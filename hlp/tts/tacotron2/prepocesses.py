import io
import json
import os
import re
import numpy as np
import tensorflow as tf

from audio_process import get_spectrograms


# 按字符切
def preprocess_sentence(s):
    s = s.lower().strip()
    s = re.sub(r"([?.!,])", r" \1 ", s)  # 切分断句的标点符号
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格
    s = re.sub(r"[^a-zA-Z?.!,]+", " ", s)
    s = s.strip()
    return s


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


# 按字符切分
def process_text_word(sentence_list):
    en_sentences = [preprocess_sentence(s) for s in sentence_list]
    return en_sentences


# 按字母切分
def process_text(sentence_list):
    sentences_list2 = []
    for s in sentence_list:
        a = ""
        s = s.lower().strip()  # 大写转小写
        s = re.sub(r"[^a-zA-Z?.!,]+", " ", s)  # 替换一些不常规符号
        s = re.sub(r'[" "]+', " ", s)  # 合并多个空格
        sentence = s.strip()  # 去掉前后空格
        for q in sentence:
            a = a + ' ' + q
        sentences_list2.append(a)
    return sentences_list2


def tokenize(texts, save_path, name):
    if name == "train":
        # 准备train之前要保存字典
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='UNK')  # 无过滤字符
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列
        # 保存字典
        json_string = tokenizer.to_json()
        with open(save_path, 'w') as f:
            json.dump(json_string, f)
        vocab_size = len(tokenizer.word_index) + 1
        sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
        return sequences, vocab_size
    else:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')  # 无过滤字符
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列
        # print(sequences[-1])
        sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
        vocab_size = len(tokenizer.word_index) + 1
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
    vocab_size = len(tokenizer.word_index)
    return tokenizer, vocab_size


# 训练的时候对取得的句子列表处理
def dataset_txt(sentence_list, save_path, name):
    en = process_text(sentence_list)
    en_seqs, vocab_size = tokenize(en, save_path, name)
    return en_seqs, vocab_size


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
    for i in range(len(mel_len_wav)):
        j = mel_len_wav[i]
        tar_token[i, (j - 1):] = 1
    return tar_token


def create_dataset(batch_size, input_ids, mel_gts, tar_token):
    BUFFER_SIZE = len(input_ids)
    steps_per_epoch = BUFFER_SIZE // batch_size
    # dataset = tf.data.Dataset.from_tensor_slices((input_ids, mel_gts)).shuffle(BUFFER_SIZE)
    dataset = tf.data.Dataset.from_tensor_slices((input_ids, mel_gts, tar_token)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset, steps_per_epoch


if __name__ == '__main__':
    pass
