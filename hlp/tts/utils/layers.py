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
