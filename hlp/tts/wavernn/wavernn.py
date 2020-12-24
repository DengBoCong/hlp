import tensorflow as tf
import numpy as np

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, dims):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=dims, kernel_size=1, use_bias=False)
        self.conv2 = tf.keras.layers.Conv1D(filters=dims, kernel_size=1, use_bias=False)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

    def call(self, x):

        residual = x
        print("residual:", residual.shape)
        print("x1:", x.shape)

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)


        return x + residual


class MelResNet(tf.keras.Model):
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad):
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = tf.keras.layers.Conv1D(compute_dims, kernel_size=k_size, use_bias=False)
        self.batch_norm = tf.keras.layers.BatchNormalization()

        self.layer = []
        #self.layer = tf.keras.Sequential()
        for i in range(res_blocks):
            self.layer.append(ResBlock(compute_dims))
        self.conv_out = tf.keras.layers.Conv1D(res_out_dims, kernel_size=1)

    def call(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = tf.nn.relu(x)
        print("x:::", x.shape)
        for f in self.layer:
            x = f(x)
            print(1)
        print(2)
        x = self.conv_out(x)
        return x


class Stretch2d(tf.keras.layers.Layer):
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def call(self, x):
        b, c, h, w = x.shape

        x = tf.expand_dims(x, axis=-1)
        x = tf.expand_dims(x, axis=3)

        x = tf.tile(x, [1, 1, 1, self.y_scale, 1, self.x_scale])
        return tf.reshape(x, (b, c, h * self.y_scale, w * self.x_scale))


class UpsampleNetwork(tf.keras.layers.Layer):
    def __init__(self, feat_dims, upsample_scales, compute_dims,
                 res_blocks, res_out_dims, pad):
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = []
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = tf.keras.layers.Conv2D(1, kernel_size=k_size,
                                          kernel_initializer=tf.constant_initializer(1. / k_size[1]),
                                          padding="same", use_bias=False)
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def call(self, m):
        aux = self.resnet(m)
        aux = tf.expand_dims(aux, axis=1)

        aux = self.resnet_stretch(aux)
        aux = tf.squeeze(aux, axis=1)
        m = tf.expand_dims(m, axis=1)
        for f in self.up_layers:
            m = f(m)
        m = tf.squeeze(m, axis=1)[:, :, self.indent:-self.indent]
        return tf.transpose(m, (0, 2, 1)), tf.transpose(aux, (0, 2, 1))


class WaveRNN(tf.keras.Model):
    def __init__(self, rnn_dims, fc_dims, bits, pad, upsample_factors,
                 feat_dims, compute_dims, res_out_dims, res_blocks,
                 hop_length, sample_rate, mode='RAW'):
        super().__init__()
        self.mode = mode
        self.pad = pad
        if self.mode == 'RAW':
            self.n_classes = 2 ** bits
        elif self.mode == 'MOL':
            self.n_classes = 30
        else:
            RuntimeError("Unknown model mode value - ", self.mode)

        # List of rnns to call `flatten_parameters()` on
        self._to_flatten = []

        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims, res_blocks, res_out_dims, pad)
        self.I = tf.keras.layers.Dense(rnn_dims, activation=None)

        self.rnn1 = tf.keras.layers.GRU(rnn_dims)
        self.rnn2 = tf.keras.layers.GRU(rnn_dims)
        self._to_flatten += [self.rnn1, self.rnn2]

        self.fc1 = tf.keras.layers.Dense(fc_dims, activation=None)
        self.fc2 = tf.keras.layers.Dense(fc_dims, activation=None)
        self.fc3 = tf.keras.layers.Dense(self.n_classes, activation=None)

    def call(self, x, mels):

        #self.step += 1
        bsize = x.shape[0]
        h1 = tf.zeros(1, bsize, self.rnn_dims)
        h2 = tf.zeros(1, bsize, self.rnn_dims)
        mels, aux = self.upsample(mels)

        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]


        print("x:", x.shape)
        print("mels:", mels.shape)
        print("a1:", a1.shape)
        exit(0)
        x = tf.concat([tf.expand_dims(x, axis=-1), mels, a1], axis=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = tf.concat([x, a2], axis=2)
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = tf.concat([x, a3], axis=2)
        x = tf.nn.relu(self.fc1(x))

        x = tf.concat([x, a4], axis=2)
        x = tf.nn.relu(self.fc2(x))
        return self.fc3(x)

