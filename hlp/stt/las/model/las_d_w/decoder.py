# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:21:38 2020
a 2-layer RNN decoder of cell dimension w
The attention vectors are fed into a 2-layer RNN decoder of cell dimension w,
which yields the tokens for the transcript. 
@author: 九童
"""
import tensorflow as tf
from hlp.utils.layers import BahdanauAttention


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, w):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn1 = tf.keras.layers.LSTM(w)
        self.rnn2 = tf.keras.layers.LSTM(w, return_state=True)
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

        # 将合并后的向量传送到 RNN
        x = self.rnn1(x)
        output, state = self.rnn2(x)

        # 输出的形状 == （批大小 * 1，隐藏层大小）
        output = tf.reshape(output, (-1, output.shape[2]))

        # 输出的形状 == （批大小，vocab）
        x = self.fc(output)

        return x, state, attention_weights
