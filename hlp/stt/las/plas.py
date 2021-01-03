import tensorflow as tf
from hlp.utils.layers import BahdanauAttention


class PBLSTM(tf.keras.layers.Layer):
    """金字塔BiLSTM

    逐层缩减序列长度
    """
    def __init__(self, dim):
        super(PBLSTM, self).__init__()
        self.dim = dim
        self.bidi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.dim, return_sequences=True))

    @tf.function
    def call(self, inputs):
        y = self.bidi_lstm(inputs)

        if tf.shape(inputs)[1] % 2 == 1:
            y = tf.keras.layers.ZeroPadding1D(padding=(0, 1))(y)

        y = tf.keras.layers.Reshape(target_shape=(-1, int(self.dim * 4)))(y)
        return y


class Encoder(tf.keras.Model):
    def __init__(self, dim, enc_units):
        # TODO: 金字塔层数可变
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.dim = dim
        # Listen; Lower resoultion by 8x
        self.plstm1 = PBLSTM(self.dim // 2)
        self.plstm2 = PBLSTM(self.dim // 2)
        self.plstm3 = PBLSTM(self.dim // 2)

    def call(self, x):
        """声学特征序列编码

        :param x: 声学特征序列
        :return: 缩减后的编码特征序列
        """
        x = self.plstm1(x)
        x = self.plstm2(x)
        output = self.plstm3(x)
        return output


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(dec_units)

    def call(self, x, hidden, enc_output):
        """解码

        :param x: 目标符号, （批大小，id）
        :param hidden: 解码器状态, （批大小，隐藏层大小）
        :param enc_output: 编码器输出, （批大小，最大长度，隐藏层大小）
        :return: token分布, 解码器专题, 注意力权重
        """
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
        x = self.embedding(x)

        # x 在拼接 （concatenation） 后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)
        # 输出的形状 == （批大小 * 1，隐藏层大小）
        output = tf.reshape(output, (-1, output.shape[2]))

        # 输出的形状 == （批大小，vocab）
        tokens_prob = self.fc(output)

        return tokens_prob, state, attention_weights


class PLAS(tf.keras.Model):
    def __init__(self, vocab_tar_size, embedding_dim, units, batch_size):
        super(PLAS, self).__init__()
        self.units = units
        self.batch_size = batch_size
        # TODO: 编码器和解码器使用不同的单元数
        self.encoder = Encoder(embedding_dim, units, batch_size)
        self.decoder = Decoder(vocab_tar_size, embedding_dim, units)

    def call(self, x, enc_hidden, dec_input):
        """

        :param x: 编码器输入
        :param enc_hidden:
        :param dec_input: 解码器输入
        :return: 解码器预测, 解码器状态
        """
        enc_output = self.encoder(x)
        dec_hidden = enc_hidden  # 编码器状态作为解码器初始状态？
        predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
        return predictions, dec_hidden

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.units))


if __name__ == "__main__":
    import numpy as np
    # a = np.arange(12).reshape((1, 4, 3)).astype(np.float)
    a = np.arange(15).reshape((1, 5, 3)).astype(np.float)
    p_lstm = PBLSTM(8)
    r = p_lstm(a)
    print(r.shape)
