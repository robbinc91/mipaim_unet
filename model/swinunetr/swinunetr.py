import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import argparse
import nibabel as nib
from scipy.ndimage import zoom

# Swin Transformer Block (channels-first, with shape debugging)
class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, window_size=4, shift_size=0, mlp_ratio=4.0, drop_rate=0.0):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        print('Create SwintransformerBlock', 'dim:', dim, "mlp_ratio:", mlp_ratio)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6, axis=1)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, axis=1)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads, dropout=drop_rate)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = models.Sequential([
            layers.Dense(mlp_hidden_dim, activation='gelu'),
            layers.Dropout(drop_rate),
            layers.Dense(dim),
            layers.Dropout(drop_rate)
        ])

    def call(self, x, training=False):
        # Input shape: (batch, channels, height, width, depth)
        B, C, H, W, D = x.shape
        assert H % self.window_size == 0, f"Height {H} must be divisible by window_size {self.window_size}"
        assert W % self.window_size == 0, f"Width {W} must be divisible by window_size {self.window_size}"
        assert D % self.window_size == 0, f"Depth {D} must be divisible by window_size {self.window_size}"

        # Window partitioning
        if self.shift_size > 0:
            x = tf.roll(x, shift=[-self.shift_size, -self.shift_size, -self.shift_size], axis=[2, 3, 4])

        # Reshape for windowed attention
        num_windows = (H // self.window_size) * (W // self.window_size) * (D // self.window_size)
        windowed_shape = (B * num_windows, self.window_size * self.window_size * self.window_size, C)
        x = tf.transpose(x, perm=[0, 2, 3, 4, 1])  # To NHWDC
        x = tf.reshape(x, windowed_shape)

        # Attention
        x = self.norm1(x)
        x = self.attn(x, x, training=training)
        
        # Reverse window partitioning
        x = tf.reshape(x, (B, H, W, D, C))
        x = tf.transpose(x, perm=[0, 4, 1, 2, 3])  # Back to NCHWD

        if self.shift_size > 0:
            x = tf.roll(x, shift=[self.shift_size, self.shift_size, self.shift_size], axis=[2, 3, 4])

        # FFN
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm, training=training)

        print(x_norm.shape)
        print(x.shape, B, C, H, W, D)
        print(mlp_out.shape, B, C, H, W, D)

        tf.debugging.assert_shapes([(x, (B, C, H, W, D)), (mlp_out, (B, C, H, W, D))])
        x = x + mlp_out

        return x

# Basic Convolutional Block (channels-first)
class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1):
        super(ConvBlock, self).__init__()
        self.conv = layers.Conv3D(filters, kernel_size, strides=strides, padding='same', data_format='channels_first')
        self.norm = layers.BatchNormalization(axis=1)
        self.relu = layers.ReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.norm(x, training=training)
        x = self.relu(x)
        return x

# Swin UNETR Model (channels-first)
class SwinUNETR(models.Model):
    def __init__(self, img_size=(128, 128, 128), in_channels=4, out_channels=3, feature_size=48):
        super(SwinUNETR, self).__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size

        # Patch embedding
        self.patch_embed = layers.Conv3D(feature_size, kernel_size=2, strides=2, padding='same', data_format='channels_first')

        # Encoder
        self.encoder1 = ConvBlock(feature_size)
        self.encoder2 = SwinTransformerBlock(dim=feature_size, num_heads=4, window_size=4)
        self.encoder3 = SwinTransformerBlock(dim=feature_size * 2, num_heads=8, window_size=4)
        self.encoder4 = SwinTransformerBlock(dim=feature_size * 4, num_heads=16, window_size=4)
        self.encoder5 = SwinTransformerBlock(dim=feature_size * 8, num_heads=32, window_size=4)

        # Downsampling
        self.downsample2 = layers.MaxPooling3D(pool_size=2, data_format='channels_first', padding='same')
        self.downsample3 = layers.MaxPooling3D(pool_size=2, data_format='channels_first', padding='same')
        self.downsample4 = layers.MaxPooling3D(pool_size=2, data_format='channels_first', padding='same')

        # Decoder
        self.decoder4 = layers.Conv3DTranspose(feature_size * 4, kernel_size=2, strides=2, padding='same', data_format='channels_first')
        self.decoder3 = layers.Conv3DTranspose(feature_size * 2, kernel_size=2, strides=2, padding='same', data_format='channels_first')
        self.decoder2 = layers.Conv3DTranspose(feature_size, kernel_size=2, strides=2, padding='same', data_format='channels_first')
        self.decoder1 = ConvBlock(feature_size)

        # Final output
        self.final_conv = layers.Conv3D(out_channels, kernel_size=1, activation='softmax', data_format='channels_first')

    def call(self, x, training=False):
        # Input shape: (batch, in_channels, height, width, depth)
        x1 = self.encoder1(x, training=training)
        x = self.patch_embed(x)
        x2 = self.encoder2(x, training=training)
        x = self.downsample2(x2)
        x3 = self.encoder3(x, training=training)
        x = self.downsample3(x3)
        x4 = self.encoder4(x, training=training)
        x = self.downsample4(x4)
        x5 = self.encoder5(x, training=training)

        # Decoder with skip connections
        d4 = self.decoder4(x5, training=training)
        d4 = tf.concat([d4, x4], axis=1)
        d3 = self.decoder3(d4, training=training)
        d3 = tf.concat([d3, x3], axis=1)
        d2 = self.decoder2(d3, training=training)
        d2 = tf.concat([d2, x2], axis=1)
        d1 = self.decoder1(d2, training=training)

        # Output
        out = self.final_conv(d1)
        return out

# Inference function (channels-first)
def inference(model, image, crop_size=(128, 128, 128)):
    """Perform sliding window inference for 3D image."""
    def sliding_window(image, step_size, window_size):
        patches = []
        coords = []
        for z in range(0, image.shape[2] - window_size[2] + 1, step_size):
            for y in range(0, image.shape[1] - window_size[1] + 1, step_size):
                for x in range(0, image.shape[0] - window_size[0] + 1, step_size):
                    patch = image[x:x+window_size[0], y:y+window_size[1], z:z+window_size[2]]
                    patches.append(patch)
                    coords.append((x, y, z))
        return np.array(patches), coords

    # Preprocess image (assume image is HWD, add batch and channels)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize
    image = np.transpose(image, (3, 0, 1, 2))  # To CHWD
    image = np.expand_dims(image, axis=0)  # Add batch: NCHWD

    # Sliding window inference
    step_size = crop_size[0] // 2
    patches, coords = sliding_window(np.transpose(image[0], (1, 2, 3, 0)), step_size, crop_size)  # To HWDC
    patches = np.transpose(patches, (0, 4, 1, 2, 3))  # To NCHWD
    predictions = model.predict(patches, batch_size=1)

    # Aggregate predictions
    output = np.zeros((model.out_channels,) + image.shape[2:])  # CHWD
    count_map = np.zeros(image.shape[2:])  # HWD
    for pred, (x, y, z) in zip(predictions, coords):
        output[:, x:x+crop_size[0], y:y+crop_size[1], z:z+crop_size[2]] += pred
        count_map[x:x+crop_size[0], y:y+crop_size[1], z:z+crop_size[2]] += 1
    output = output / (count_map[np.newaxis, ...] + 1e-6)
    return output

# Command-line interface for inference
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task_form', type=str, required=True, help="task form")
    parser.add_argument('-o', '--model_out_channels', type=int, required=True, help="model out channels")
    parser.add_argument('-w', '--model_weights_path', type=str, required=True, help="model weights path")
    parser.add_argument('-d', '--data_path', type=str, required=True, help="data path")
    parser.add_argument('-c', '--crop', type=str, required=True, help="crop")
    parser.add_argument('-p', '--prediction_path', type=str, required=True, help="prediction path")
    parser.add_argument('-v', '--visualization_path', type=str, required=True, help="visualization path")
    args = parser.parse_args()

    # Load model
    model = SwinUNETR(img_size=(128, 128, 128), in_channels=4, out_channels=args.model_out_channels)
    model.load_weights(args.model_weights_path)

    # Load and preprocess image (NIfTI, assuming HWDC input)
    img = nib.load(args.data_path).get_fdata()
    crop_size = tuple(map(int, args.crop.split(',')))

    # Perform inference
    pred = inference(model, img, crop_size=crop_size)

    # Save prediction (transpose to HWDC for NIfTI)
    pred = np.transpose(pred, (1, 2, 3, 0))
    nib.save(nib.Nifti1Image(pred, np.eye(4)), args.prediction_path)

    # Save visualization (simple slice visualization)
    import matplotlib.pyplot as plt
    plt.imsave(args.visualization_path, pred[..., pred.shape[-1]//2, 0], cmap='gray')
    plt.close()

if __name__ == "__main__":
    main()