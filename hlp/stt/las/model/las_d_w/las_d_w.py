# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 11:34:26 2020
用于SpecAugment的las_d_w模型
@author: 九童
"""
import tensorflow as tf
import numpy as np
from encoder import Encoder
from decoder import Decoder


class las_d_w_model(tf.keras.Model):
    def __init__(self, vocab_tar_size, d, w, embedding_dim, dec_units, batch_size):
        super(las_d_w_model, self).__init__()
        self.vocab_tar_size = vocab_tar_size
        self.d = d
        self.w = w
        self.batch_size = batch_size
        self.encoder = Encoder(d, w, batch_size)
        self.decoder = Decoder(vocab_tar_size, embedding_dim, dec_units, w)

    def call(self, inputx_1, enc_hidden, dec_input):
        enc_output, enc_hidden = self.encoder(inputx_1, enc_hidden)  # 前向计算，编码
        dec_hidden = enc_hidden  # 编码器状态作为解码器初始状态？
        predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
        return predictions, dec_hidden

    def initialize_hidden_state(self):
        return self.encoder.initialize_hidden_state()



