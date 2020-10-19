# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:54:55 2020

@author: 九童
语音识别模型的使用和评估
"""
import numpy as np
import scipy.io.wavfile
import tensorflow as tf

from hlp.stt.las.data_processing import mfcc_extract


def evaluate(wav_path, max_length_inp, max_length_targ, targ_lang_tokenizer, model):
    # sentence = preprocess_en_sentence(sentence)
    sample_rate, signal = scipy.io.wavfile.read(wav_path)
    wav_mfcc = mfcc_extract.MFCC(sample_rate, signal)

    # inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]  # token编码
    # inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
    # maxlen=max_length_inp,
    # padding='post')  # 填充
    # print('====wav_mfcc.shape = {}'.format(wav_mfcc.shape))#(60, 39)
    wav_mfcc = tf.expand_dims(wav_mfcc, 0)  # (1,60, 39)
    wav_mfcc = tf.keras.preprocessing.sequence.pad_sequences(wav_mfcc, maxlen=max_length_inp, padding='post',
                                                             dtype=float)
    wav_mfcc = tf.convert_to_tensor(wav_mfcc)  # numpy数组转换成张量

    # print('====wav_mfcc.shape = {}'.format(wav_mfcc.shape))#(1,93, 39)
    result = ''  # 语音识别结果字符串

    dec_input = tf.keras.utils.to_categorical([targ_lang_tokenizer.word_index['<start>'] - 1],
                                              num_classes=len(targ_lang_tokenizer.word_index) + 1)
    dec_input = tf.expand_dims(dec_input, 1)
    dec_input = np.array(dec_input).astype(int)
    dec_input = tf.convert_to_tensor(dec_input)
    # print('====dec_input = {}'.format(dec_input))
    for t in range(max_length_targ):  # 逐步解码或预测
        predictions = model([wav_mfcc, dec_input])
        # print('====predictions.shape = {}'.format(predictions.shape))
        predicted_id = tf.argmax(predictions[0][0]).numpy() + 1  # 贪婪解码，取最大
        # print('====predicted_id = {}'.format(predicted_id))

        result += targ_lang_tokenizer.index_word[predicted_id] + ' '  # 目标句子
        # print('====result = {}'.format(result))
        if targ_lang_tokenizer.index_word[predicted_id] == '<end>':
            return result

        # 预测的 ID 被输送回模型
        dec_input = tf.keras.utils.to_categorical([predicted_id - 1],
                                                  num_classes=len(targ_lang_tokenizer.word_index) + 1)
        dec_input = tf.expand_dims(dec_input, 1)
        # print('====afterdec_input.shape = {}'.format(dec_input.shape))

    return result


def speech_recognition(wav_path, max_length_inp, max_length_targ, targ_lang_tokenizer, model):
    result = evaluate(wav_path, max_length_inp, max_length_targ, targ_lang_tokenizer, model)
    print('Predicted speech_recognition: {}'.format(result))
