import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K

# Set channels-first data format
tf.keras.backend.set_image_data_format('channels_first')

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Patch Partitioning Layer
class PatchPartition(layers.Layer):
    def __init__(self, patch_size=(2, 2, 2)):
        super().__init__()
        self.patch_size = patch_size

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        channels = inputs.shape[1]
        patches = tf.extract_volume_patches(
            inputs,
            ksizes=[1, 1, *self.patch_size],
            strides=[1, 1, *self.patch_size],
            padding='VALID'
        )
        patch_dims = tf.reduce_prod(self.patch_size) * channels
        return tf.reshape(patches, [batch_size, -1, patch_dims])

# Patch Merging Layer
class PatchMerging(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.norm = layers.LayerNormalization(epsilon=1e-6, axis=-1)
        self.reduction = layers.Dense(2 * dim, use_bias=False)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        num_patches = inputs.shape[1]
        dim = inputs.shape[2]
        spatial_dim = tf.cast(tf.pow(tf.cast(num_patches // 8, tf.float32), 1/3), tf.int32)
        patches = tf.reshape(inputs, [batch_size, spatial_dim, spatial_dim, spatial_dim, 8, dim])
        patches = tf.concat([patches[..., i, :] for i in range(8)], axis=-1)
        patches = tf.reshape(patches, [batch_size, spatial_dim**3, dim * 8])
        patches = self.norm(patches)
        return self.reduction(patches)

# Window Partitioning and Reversing Functions
def window_partition(inputs, window_size, spatial_shape):
    batch_size = tf.shape(inputs)[0]
    channels = inputs.shape[-1]
    D, H, W = [tf.convert_to_tensor(dim, tf.int32) for dim in spatial_shape]
    x = tf.reshape(inputs, [batch_size, D, H, W, channels])
    pad_d = tf.math.mod(-D, window_size[0])
    pad_h = tf.math.mod(-H, window_size[1])
    pad_w = tf.math.mod(-W, window_size[2])
    if tf.reduce_any(tf.stack([pad_d, pad_h, pad_w]) > 0):
        x = tf.pad(x, [[0, 0], [0, pad_d], [0, pad_h], [0, pad_w], [0, 0]])
    Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w
    x = tf.reshape(x, [
        batch_size,
        Dp // window_size[0], window_size[0],
        Hp // window_size[1], window_size[1],
        Wp // window_size[2], window_size[2],
        channels
    ])
    x = tf.transpose(x, [0, 1, 3, 5, 2, 4, 6, 7])
    windows = tf.reshape(x, [-1, window_size[0] * window_size[1] * window_size[2], channels])
    return windows, (Dp, Hp, Wp)

def window_reverse(windows, window_size, spatial_shape, padded_shape):
    D, H, W = [tf.convert_to_tensor(dim, tf.int32) for dim in spatial_shape]
    Dp, Hp, Wp = [tf.convert_to_tensor(dim, tf.int32) for dim in padded_shape]
    num_windows = (Dp // window_size[0]) * (Hp // window_size[1]) * (Wp // window_size[2])
    batch_size = tf.shape(windows)[0] // num_windows
    x = tf.reshape(windows, [
        batch_size,
        Dp // window_size[0], Hp // window_size[1], Wp // window_size[2],
        window_size[0], window_size[1], window_size[2],
        -1
    ])
    x = tf.transpose(x, [0, 1, 4, 2, 5, 3, 6, 7])
    x = tf.reshape(x, [batch_size, Dp, Hp, Wp, -1])
    x = x[:, :D, :H, :W, :]
    num_patches = tf.reduce_prod([D, H, W])
    return tf.reshape(x, [batch_size, num_patches, tf.shape(x)[-1]])

# Window Attention Layer
class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.dim_per_head = dim // num_heads
        self.scale = self.dim_per_head ** -0.5
        self.qkv = layers.Dense(dim * 3)
        self.proj = layers.Dense(dim)
        self.softmax = layers.Softmax(axis=-1)
        table_shape = ((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        self.relative_position_bias_table = self.add_weight(
            name='relative_position_bias_table',
            shape=table_shape,
            initializer='zeros',
            trainable=True
        )
        coords = tf.stack(tf.meshgrid(
            tf.range(window_size[0]),
            tf.range(window_size[1]),
            tf.range(window_size[2]),
            indexing='ij'
        ))
        coords_flatten = tf.reshape(coords, [3, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.transpose(relative_coords, [1, 2, 0])
        relative_coords = relative_coords + [window_size[0] - 1, window_size[1] - 1, window_size[2] - 1]
        relative_coords = relative_coords * [
            (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
            2 * window_size[2] - 1,
            1
        ]
        self.relative_position_index = tf.reduce_sum(relative_coords, axis=-1)

    def call(self, inputs, spatial_shape, mask=None):
        batch_size = tf.shape(inputs)[0]
        windows, padded_shape = window_partition(inputs, self.window_size, spatial_shape)
        num_windows = tf.shape(windows)[0] // batch_size
        qkv = self.qkv(windows)
        qkv = tf.reshape(qkv, [tf.shape(windows)[0], -1, 3, self.num_heads, self.dim_per_head])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(self.relative_position_index, [-1])
        )
        relative_position_bias = tf.reshape(relative_position_bias, [
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1
        ])
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, 0)
        if mask is not None:
            attn = tf.reshape(attn, [batch_size, num_windows, self.num_heads, -1, -1])
            attn = attn + tf.expand_dims(tf.expand_dims(mask, 0), 2)
            attn = tf.reshape(attn, [-1, self.num_heads, -1, -1])
        attn = self.softmax(attn)
        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [tf.shape(windows)[0], -1, self.dim])
        out = self.proj(out)
        out = window_reverse(out, self.window_size, spatial_shape, padded_shape)
        return out

# Swin Transformer Block
class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, window_size, num_heads, shift_size=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size or (0, 0, 0)
        self.num_heads = num_heads
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = models.Sequential([
            layers.Dense(dim * 4, activation='gelu'),
            layers.Dense(dim)
        ])

    def create_window_mask(self, spatial_shape):
        return None

    def call(self, inputs, spatial_shape):
        batch_size = tf.shape(inputs)[0]
        shortcut = inputs
        x = self.norm1(inputs)
        if any(s > 0 for s in self.shift_size):
            x = tf.reshape(x, [batch_size, *spatial_shape, -1])
            x = tf.roll(x, shift=[-s for s in self.shift_size], axis=(1, 2, 3))
            x = tf.reshape(x, [batch_size, -1, self.dim])
        mask = self.create_window_mask(spatial_shape)
        x = self.attn(x, spatial_shape, mask=mask)
        if any(s > 0 for s in self.shift_size):
            x = tf.reshape(x, [batch_size, *spatial_shape, -1])
            x = tf.roll(x, shift=self.shift_size, axis=(1, 2, 3))
            x = tf.reshape(x, [batch_size, -1, self.dim])
        x = shortcut + x
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        return shortcut + x

# SwinUNETR Model
def SwinUNETR(input_shape=(1, 80, 80, 96), num_classes=4, embed_dim=16, depths=[2, 2, 2], num_heads=[3, 4, 6, 12], window_size=(4, 4, 4)):
    inputs = layers.Input(shape=input_shape)
    x = PatchPartition(patch_size=(2, 2, 4))(inputs)
    x = layers.Dense(embed_dim)(x)
    spatial_shapes = [
        (input_shape[1] // 4, input_shape[2] // 2, input_shape[3] // 4),
        (input_shape[1] // 4, input_shape[2] // 4, input_shape[3] // 2),
        (input_shape[1] // 8, input_shape[2] // 8, input_shape[3] // 2),
        (input_shape[1] // 16, input_shape[2] // 16, input_shape[3] // 2)
    ]
    skips = []
    for i, (depth, heads) in enumerate(zip(depths, num_heads)):
        for j in range(depth):
            shift_size = (window_size[0] // 4, window_size[1] // 2, window_size[2] // 2) if j % 2 == 1 else (0, 0, 2)
            x = SwinTransformerBlock(embed_dim * (2 ** i), window_size, heads, shift_size)(x, spatial_shapes[i])
        skips.append(x)
        if i < len(depths) - 1:
            x = PatchMerging(dim=embed_dim * (2 ** i))(x)

    # Decoder
    for i in range(len(depths) - 1, -1, -1):
        if i > 0:
            spatial_shape = spatial_shapes[i - 1]   # Pre-upsampled shape
            num_patches_stage = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
            x = layers.Dense(num_patches_stage * embed_dim * (2 ** i))(x)
            x = tf.reshape(x, [tf.shape(x)[0], embed_dim * (2 ** i), *spatial_shape])
            x = layers.Conv3DTranspose(64, embed_dim * (2 ** (i - 1)), 2, strides=(2, 2, 2), padding='same', data_format='channels_first')(x)
            skip = skips[i - 1]
            skip = tf.reshape(skip, [tf.shape(skip)[0], embed_dim * (2 ** i), *spatial_shape])
            skip = layers.Conv3DTranspose(embed_dim * (2 ** i), 2, strides=(2, 2, 2), padding='same', data_format='channels_first')(skip)
            x = layers.Concatenate(axis=1)([x, skip])
            x = layers.Conv3D(embed_dim * (2 ** (i - 1)), 3, padding='same', activation='gelu', data_format='channels_first')(x)
            x = tf.reshape(x, [tf.shape(x)[0], -1, embed_dim * (2 ** (i - 1))])

    # Final upsampling
    final_shape = input_shape[1:]
    x = layers.Dense(tf.reduce_prod([f * final_shape]) * embed_dim)(x)
    x = tf.reshape(x, [tf.shape(x)[0], embed_dim, *final_shape])
    x = layers.Conv3D(num_classes, 1, activation='softmax', data_format='channels_first')(x)
    return models.Model(inputs, x)

# Create and compile the model
def create_swinunetr():
    model = SwinUNETR(input_shape=(1, 80, 80, 30), num_classes=2)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Example usage
if __name__ == "__main__":
    model = create_swinunetr()
    model.build((None, 1, 80, 80, 30))
    input_tensor = tf.random.normal((1, 1, 80, 80, 30), dtype=tf.float32)
    output = model(input_tensor)
    print("Output shape:", output.shape)
