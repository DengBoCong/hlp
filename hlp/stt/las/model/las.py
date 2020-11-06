# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:59:23 2020

@author: 九童
LAS模型
"""
import tensorflow as tf
from hlp.stt.las.model import encoder
from hlp.stt.las.model import decoder

class las_model(tf.keras.Model):
    def __init__(self, vocab_tar_size,n_mfcc, embedding_dim, units, BATCH_SIZE):
        super(las_model, self).__init__()
        self.vocab_tar_size = vocab_tar_size
        self.n_mfcc = n_mfcc
        self.embedding_dim = embedding_dim
        self.units = units
        self.BATCH_SIZE = BATCH_SIZE
        self.encoder = encoder.Encoder(n_mfcc, embedding_dim, units, BATCH_SIZE)
        self.decoder = decoder.Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
        
    def call(self, inputx_1,enc_hidden,dec_input):
        enc_output, enc_hidden = self.encoder(inputx_1, enc_hidden)  # 前向计算，编码
        dec_hidden = enc_hidden  # 编码器状态作为解码器初始状态？
        predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
        return predictions,dec_hidden
    
    def initialize_hidden_state(self):
        return self.encoder.initialize_hidden_state()




