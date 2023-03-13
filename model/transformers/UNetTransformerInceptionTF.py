import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50


class InceptionBlock(tf.keras.Model):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = layers.Conv3D(out_channels, kernel_size=1)
        self.conv3 = layers.Conv3D(out_channels, kernel_size=3, padding='same')
        self.conv5 = layers.Conv3D(out_channels, kernel_size=5, padding='same')
        self.pool = layers.MaxPool3D(pool_size=3, strides=1, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, x):
        conv1 = self.conv1(x)
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        pool = self.pool(x)
        out = tf.concat([conv1, conv3, conv5, pool], axis=4)
        out = self.bn(out)
        out = tf.nn.relu(out)
        return out


class UpsampleBlock(tf.keras.Model):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = layers.UpSampling3D(size=2)
        self.conv = layers.Conv3D(out_channels, kernel_size=3, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, x, skip):
        x = self.upsample(x)
        x = tf.concat([x, skip], axis=4)
        x = self.conv(x)
        x = self.bn(x)
        x = tf.nn.relu(x)
        return x


class UNetTransformerInception(tf.keras.Model):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Rescale input to 3 channels
        self.rescale = tf.keras.Sequential([
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv3D(3, kernel_size=1)
        ])

        # UNet encoding path with Inception blocks
        self.inc1 = InceptionBlock(3, 32)
        self.pool1 = layers.MaxPool3D(pool_size=2, strides=2)
        self.inc2 = InceptionBlock(32, 64)
        self.pool2 = layers.MaxPool3D(pool_size=2, strides=2)
        self.inc3 = InceptionBlock(64, 128)
        self.pool3 = layers.MaxPool3D(pool_size=2, strides=2)
        self.inc4 = InceptionBlock(128, 256)
        self.pool4 = layers.MaxPool3D(pool_size=2, strides=2)

        # Transformer
        resnet = ResNet50(include_top=False, pooling=None,
                          input_shape=(None, None, None, 256))
        self.transformer = tf.keras.Sequential(resnet.layers[:-2])

        # UNet decoding path with skip connections and transformer output
        self.up5 = UpsampleBlock(256, 128, 256)
        self.up6 = UpsampleBlock(256, 64, 128)
        self.up7 = UpsampleBlock(128, 32, 64)
        self.up8 = UpsampleBlock(64, 32, 32)

        # Output layer with sigmoid activation for multilabel segmentation
        self.out = layers.Conv3D(out_channels, kernel_size=1)
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        #
        pass
