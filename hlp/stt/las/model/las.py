# -*- coding: utf-8 -*-

import tensorflow as tf
from hlp.utils.layers import BahdanauAttention


class pBLSTM(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(pBLSTM, self).__init__()
        self.dim = dim
        self.LSTM = tf.keras.layers.LSTM(self.dim, return_sequences=True)
        self.bidi_LSTM = tf.keras.layers.Bidirectional(self.LSTM)

    @tf.function
    def call(self, inputs):
        y = self.bidi_LSTM(inputs)

        if tf.shape(inputs)[1] % 2 == 1:
            y = tf.keras.layers.ZeroPadding1D(padding=(0, 1))(y)

        y = tf.keras.layers.Reshape(target_shape=(-1, int(self.dim * 4)))(y)
        return y


class Encoder(tf.keras.Model):
    def __init__(self, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.dim = embedding_dim
        # Listen; Lower resoultion by 8x
        self.plstm1 = pBLSTM(self.dim // 2)
        self.plstm2 = pBLSTM(self.dim // 2)
        self.plstm3 = pBLSTM(self.dim // 2)

    def call(self, x):
        x = self.plstm1(x)
        x = self.plstm2(x)
        output = self.plstm3(x)
        return output

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # 用于注意力
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # 编码器输出 （enc_output） 的形状 == （批大小，最大长度，隐藏层大小）
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
        x = self.embedding(x)

        # x 在拼接 （concatenation） 后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 将合并后的向量传送到 GRU
        output, state = self.gru(x)

        # 输出的形状 == （批大小 * 1，隐藏层大小）
        output = tf.reshape(output, (-1, output.shape[2]))

        # 输出的形状 == （批大小，vocab）
        x = self.fc(output)

        return x, state, attention_weights


class las_model(tf.keras.Model):
    def __init__(self, vocab_tar_size, embedding_dim, units, batch_size):
        super(las_model, self).__init__()
        self.vocab_tar_size = vocab_tar_size
        self.embedding_dim = embedding_dim
        self.units = units
        self.batch_size = batch_size
        self.encoder = Encoder(embedding_dim, units, batch_size)
        self.decoder = Decoder(vocab_tar_size, embedding_dim, units)

    def call(self, inputx_1, enc_hidden, dec_input):
        enc_output = self.encoder(inputx_1)  # 前向计算，编码
        dec_hidden = enc_hidden  # 编码器状态作为解码器初始状态？
        predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
        return predictions, dec_hidden

    def initialize_hidden_state(self):
        return self.encoder.initialize_hidden_state()
