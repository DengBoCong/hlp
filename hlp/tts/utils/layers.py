import tensorflow as tf


class ConvDropBN(tf.keras.layers.Layer):
    """
    卷积-Dropout-BatchNormalization块
    """

    def __init__(self, filters, kernel_size, activation, dropout_rate):
        """
        :param filters: 输出空间维数
        :param kernel_size: 卷积核大小
        :param activation: 激活方法
        :param dropout_rate: dropout采样率
        """
        super(ConvDropBN, self).__init__()
        self.conv1d = tf.keras.layers.Conv1D(filters, kernel_size,
                                             padding="same", activation=activation)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        outputs = self.conv1d(inputs)
        outputs = self.dropout(outputs)
        outputs = self.norm(outputs)
        return outputs


class DecoderPreNet(tf.keras.layers.Layer):
    """
    Decoder的pre_net，用于映射频谱样本的空间
    """

    def __init__(self, pre_net_units, pre_net_layers_num, pre_net_dropout_rate):
        """
        :param pre_net_units: 全连接层单元数
        :param pre_net_layers_num: pre_net层数
        :param pre_net_dropout_rate: dropout采样率
        """
        super().__init__()
        self.pre_net_units = pre_net_units
        self.pre_net_layers_num = pre_net_layers_num
        self.pre_net_dropout_rate = pre_net_dropout_rate
        self.pre_net_dense = [
            tf.keras.layers.Dense(units=self.pre_net_units, activation='relu')
            for i in range(self.pre_net_layers_num)
        ]
        self.dropout = tf.keras.layers.Dropout(rate=self.pre_net_dropout_rate)

    def call(self, inputs):
        outputs = inputs
        for layer in self.pre_net_dense:
            outputs = layer(outputs)
            outputs = self.dropout(outputs)
        return outputs


class PostNet(tf.keras.layers.Layer):
    """
    Tacotron2的PostNet，包含n_conv_encoder数量的卷积层
    """

    def __init__(self, encoder_conv_num: int, post_net_conv_num: int, post_net_filters: int,
                 post_net_kernel_sizes: int, post_net_dropout: float,
                 post_net_activation: str, num_mel: int):
        """
        :param encoder_conv_num: encoder卷积层数量
        :param post_net_conv_num: post_net的卷积层数量
        :param post_net_filters: post_net卷积输出空间维数
        :param post_net_kernel_sizes: post_net卷积核大小
        :param post_net_dropout: post_net的dropout采样率
        :param post_net_activation: post_net卷积激活函数
        :param n_mels: 梅尔带数
        """
        super().__init__()
        self.conv_batch_norm = []
        for i in range(encoder_conv_num):
            if i == post_net_conv_num - 1:
                conv = ConvDropBN(filters=post_net_filters, kernel_size=post_net_kernel_sizes,
                                  activation=None, dropout_rate=post_net_dropout)
            else:
                conv = ConvDropBN(filters=post_net_filters, kernel_size=post_net_kernel_sizes,
                                  activation=post_net_activation, dropout_rate=post_net_dropout)
            self.conv_batch_norm.append(conv)

        self.fc = tf.keras.layers.Dense(units=num_mel, activation=None, name="frame_projection1")

    def call(self, inputs):
        x = tf.transpose(inputs, [0, 2, 1])
        for _, conv in enumerate(self.conv_batch_norm):
            x = conv(x)
        x = self.fc(x)
        x = tf.transpose(x, [0, 2, 1])
        return x
