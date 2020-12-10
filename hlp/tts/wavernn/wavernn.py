import tensorflow as tf
import numpy as np
class wavernn(tf.keras.Model):
    def __init__(self,
                 quantization_channels=256,
                 gru_channels=896,
                 fc_channels=896,
                 lc_channels=80):
        super().__init__()
        self.quantization_channels = quantization_channels
        self.gru_channels = gru_channels
        self.split_size = gru_channels // 2
        self.fc_channels = fc_channels
        self.lc_channels = lc_channels

        #它的输入维度是lc_channels + 3？
        self.gru = tf.keras.layers.GRU(self.gru_channels)
        self.dense1 = tf.keras.layers.Dense(
            units=fc_channels, activation='relu', name="frame_projection")
        self.fc_coarse = tf.keras.layers.Dense(
            units=quantization_channels, activation=None, name="frame_projection")

        self.dense2 = tf.keras.layers.Dense(
            units=fc_channels, activation='relu', name="frame_projection")
        self.fc_fine = tf.keras.layers.Dense(
            units=quantization_channels, activation=None, name="frame_projection")

        #self.register_buffer('mask', self.create_mask())#还不知道怎么替换

    def create_mask(self):
        coarse_mask = tf.concat([tf.ones(self.split_size, self.lc_channels + 2),
                                 tf.zeros(self.split_size, 1)], 1)
        i2h_mask = tf.concat([coarse_mask,
                              tf.ones(self.split_size, self.lc_channels + 3)], 0)
        return tf.concat([i2h_mask, i2h_mask, i2h_mask], 0)

    def sparse_mask(self):
        pass

    def call(self, inputs, conditions):
        x = tf.concat([conditions, inputs], -1)
        h, h_n = self.gru(x)

        h_c, h_f = tf.split(h, self.split_size, axis=2)

        o_c = self.dense1(h_c)
        o_c = self.fc_coarse(o_c)
        p_c = tf.log_softmax(o_c, axis=2)

        o_f = self.dense1(h_c)
        o_f = self.fc_coarse(o_f)
        p_f = tf.log_softmax(o_f, axis=2)

        h_n = tf.squeeze(h_n, axis=0)

        return p_c, p_f, h_n


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, dims):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            filters=dims,
            kernel_size=1,
            padding="same",
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=dims,
            kernel_size=1,
            padding="same",
        )
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(tf.keras.layers.Layer):
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad):
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = tf.keras.layers.Conv1D(compute_dims, kernel_size=k_size, padding="same")
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.layers = []
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = tf.keras.layers.Conv1D(res_out_dims, kernel_size=1, padding="same")

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = tf.nn.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(tf.keras.layers.Layer):
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = tf.expand_dims(x, axis=-1)
        x = tf.expand_dims(x, axis=3)
        x = np.repeat(x, self.y_scale, axis=3)
        x = np.repeat(x, self.x_scale, axis=5)
        return tf.reshape(x, (b, c, h * self.y_scale, w * self.x_scale))

#
# class UpsampleNetwork(tf.keras.layers.Layer):
#     def __init__(self, feat_dims, upsample_scales, compute_dims,
#                  res_blocks, res_out_dims, pad):
#         super().__init__()
#         total_scale = np.cumproduct(upsample_scales)[-1]
#         self.indent = pad * total_scale
#         self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
#         self.resnet_stretch = Stretch2d(total_scale, 1)
#         self.up_layers = []
#         for scale in upsample_scales:
#             k_size = (1, scale * 2 + 1)
#             padding = (0, scale)
#             stretch = Stretch2d(scale, 1)
#             conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
#             conv.weight.data.fill_(1. / k_size[1])
#             self.up_layers.append(stretch)
#             self.up_layers.append(conv)
#
#     def forward(self, m):
#         aux = self.resnet(m).unsqueeze(1)
#         aux = self.resnet_stretch(aux)
#         aux = aux.squeeze(1)
#         m = m.unsqueeze(1)
#         for f in self.up_layers: m = f(m)
#         m = m.squeeze(1)[:, :, self.indent:-self.indent]
#         return m.transpose(1, 2), aux.transpose(1, 2)

