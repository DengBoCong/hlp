import io
import json
import os
import re
import numpy as np
import tensorflow as tf

from audio_process import get_spectrograms




# 提取语音文件名
def process_wav_name(wav_path, a):
    datanames = os.listdir(wav_path)
    wav_name_list = []
    # a=1代表它是number数据集
    if a == 1:
        for i in datanames:
            # i[:11]，11代表的是文件名字的长度，方便去csv文本文件中索引对应内容
            wav_name_list.append(i[:11])
        return wav_name_list
    else:
        for i in datanames:
            wav_name_list.append(i[:10])
        return wav_name_list


# 用语音文件名映射保存文本
def map_to_text(path_to_csv, wav_name_list):
    lines = io.open(path_to_csv, encoding='UTF-8').read().strip().split('\n')
    number = [l.split('|')[0] for l in lines[:]]
    index_sentences = [l.split('|')[1] for l in lines[:]]
    sentence_list = []
    for i in wav_name_list:
        for j in number:
            if i == j:
                x = number.index(j)
                sentence_list.append(index_sentences[x])
    return sentence_list




def tokenize(texts, maxlen_text, save_path):
    # 准备train之前要保存字典
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='UNK')  # 无过滤字符
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列
    # 保存字典
    json_string = tokenizer.to_json()
    with open(save_path, 'w') as f:
        json.dump(json_string, f)
    vocab_size = len(tokenizer.word_index) + 1
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen_text, padding='post')
    return sequences, vocab_size


#恢复字典，用来预测
def dataset_seq(texts, tokenizer, config):
    texts = process_text(texts)
    sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=config.max_len_seq, padding='post')
    return sequences

# 提取字典
def get_tokenizer_keras(path):
    """从指定路径加载保存好的字典"""
    with open(path) as f:
        json_string = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
    vocab_size = len(tokenizer.word_index)+1
    return tokenizer, vocab_size


# 训练的时候对取得的句子列表处理
def dataset_txt(sentence_list, save_path, config):
    en = process_text(sentence_list)
    en_seqs, vocab_size = tokenize(en, config.max_len_seq, save_path)
    return en_seqs, vocab_size

def dataset_mel(path, maxlen, wav_name_list2, config):
    audio_feature_list = []
    input_length_list = []
    for file in wav_name_list2:
        logmelspec, sr = get_spectrograms(path+file+'.wav', config.preemphasis, config.n_fft, config.n_mels, config.hop_length, config.win_length, config.max_db, config.ref_db, config.top_db)
        audio_feature_list.append(logmelspec)
        input_length_list.append([len(logmelspec)])

    audio_feature_numpy = tf.keras.preprocessing.sequence.pad_sequences(
        audio_feature_list,
        maxlen=maxlen,
        dtype='float32',
        padding='post'
    )
    input_tensor = tf.convert_to_tensor(audio_feature_numpy)
    input_length = tf.convert_to_tensor(input_length_list)
    input_tensor = tf.transpose(input_tensor, [0, 2, 1])
    return input_tensor, input_length



def dataset_wave(path, config):
    mel_list = []
    mel_len_wav = []
    dirs = os.listdir(path)
    for file in dirs:
        logmelspec, sr = get_spectrograms(path + file, config.preemphasis, config.n_fft, config.n_mels, config.hop_length, config.win_length, config.max_db, config.ref_db, config.top_db)
        mel_len_wav.append(len(logmelspec))
        mel_list.append(logmelspec.tolist())

    mel_numpy = tf.keras.preprocessing.sequence.pad_sequences(mel_list, maxlen=config.max_len, padding='post',
                                                              dtype='float32')
    # print(len(mel_numpy[1000]))
    inputs = tf.convert_to_tensor(mel_numpy)
    return inputs, mel_len_wav


# 用于训练stop_token
def tar_stop_token(mel_len_wav, mel_gts, max_len):
    tar_token = np.zeros((mel_gts.shape[0], max_len))
    for i in range(len(mel_len_wav[0])):
        j = mel_len_wav[i]
        j = int(j - 1)
        tar_token[i, j:] = 1
    return tar_token



if __name__ == '__main__':
    pass
