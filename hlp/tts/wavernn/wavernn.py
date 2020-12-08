import tensorflow as tf

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

# class ResBlock(nn.Module):
#     def __init__(self, dims):
#         super().__init__()
#         self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
#         self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
#         self.batch_norm1 = nn.BatchNorm1d(dims)
#         self.batch_norm2 = nn.BatchNorm1d(dims)
# 
#     def forward(self, x):
#         residual = x
#         x = self.conv1(x)
#         x = self.batch_norm1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = self.batch_norm2(x)
#         return x + residual
