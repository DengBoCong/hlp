import os
import re
import tensorflow as tf
import librosa
import numpy as np
from config2 import Tacotron2Config
import io
import csv
import pandas as pd
#文字处理
config = Tacotron2Config()
def preprocess_sentence(s):
    s = s.lower().strip()
    s = re.sub(r"([?.!,])", r" \1 ", s)  # 切分断句的标点符号
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格
    s = re.sub(r"[^a-zA-Z?.!,]+", " ", s)
    s = s.strip()
    s = s
    return s

#数字集处理的process_text
def process_text_number(text_data_path):
    sentences_list = []
    with open(text_data_path, "r", encoding='UTF-8') as f:
        sen_list = f.readlines()
    for sentence in sen_list[:]:
        sentence = sentence.strip().lower()
        sentences_list.append(preprocess_sentence(sentence))
    return sentences_list

#LJspeech数据集的处理
def process_text(text_data_path):
    lines = io.open(text_data_path, encoding='UTF-8').read().strip().split('\n')
    en_sentences = [l.split('|')[0] for l in lines[:]]
    en_sentences = [preprocess_sentence(s) for s in en_sentences]
    return en_sentences

def tokenize(texts):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')  # 无过滤字符
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列
    #print(sequences[-1])
    sequences_length = []
    for seq in sequences:
        sequences_length.append([len(seq)])
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    return sequences,tokenizer

def dataset_csv_to_txt(path_to_file,path_all_data):
    #将csv文件转化为txt
    with open(path_to_file,
              'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        j = 0
        for row in reader:
            row[0] = row[0][11:]
            str = ''.join(row)
            #print(str)
            with open('文字预处理.txt', 'a', encoding='UTF-8') as w:
                w.write(str + '\n')
        w.close()
        f.close()
    #因为有很多异常数据所以要对txt进行处理
    with open('文字预处理.txt', 'r', encoding='UTF-8') as w:
        with open(path_all_data, 'w', encoding='UTF-8') as f:
            for row in w:
                if row[0] == 'L' and row[1] == 'J':
                    #for i in range(len(row) - 11):
                    str = row[11:]
                    f.write(str)
                else:
                    f.write(row)
            f.write('\n')
        w.close()
        f.close()
    os.remove('文字预处理.txt')
    return

#划分训练集和测试集，我取前四句话为训练集，第五第六句话为测试集
def divide(text_to_file,train_to_file,test_to_file,train_number,test_number):
    # 划分训练集部分
    with open(text_to_file, 'r', encoding='UTF-8') as w:
        with open(train_to_file, 'w', encoding='UTF-8') as f:
            i = 0
            for row in w:
                if i < train_number:
                    f.write(row)
                else:
                    break
                i = i + 1
            f.write('\n')
        w.close()
        f.close()
    #划分测试集部分
    with open(text_to_file, 'r', encoding='UTF-8') as w:
        with open(test_to_file, 'w', encoding='UTF-8') as f:
            i = 0
            for row in w:
                if i < test_number + train_number and i >= train_number:
                    f.write(row)
                elif i == test_number + train_number:
                    break
                i = i + 1
            f.write('\n')
        w.close()
        f.close()
    return

def dataset_txt(path_to_file):
    en = process_text(path_to_file)
    en_seqs, en_tokenizer = tokenize(en)
    vocab_inp_size = len(en_tokenizer.word_index) + 1  # 含填充的0
    return en_seqs,vocab_inp_size

def dataset_sentence(sentence, text_data_path):
    with open(text_data_path, "w") as f:
        f.write(sentence)
    return

#mel频谱处理
def get_spectrograms(fpath):
    #设定一些参数
    config=Tacotron2Config()
    preemphasis=config.preemphasis
    n_fft=config.n_fft
    n_mels=config.n_mels
    hop_length=config.hop_length
    win_length=config.win_length
    max_db=config.max_db
    ref_db=config.ref_db
    top_db=config.top_db
    # 加载声音文件
    y, sr = librosa.load(fpath, sr=None)
    # 裁剪
    y, _ = librosa.effects.trim(y, top_db=top_db)
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])
    # 短时傅里叶变换
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length)

    # 幅度谱
    mag = np.abs(linear)  # (1+n_fft//2, T)
    # mel频谱
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)
    # 转置
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
    return mel, mag

def dataset_wave(path, config):
    mel_list = []
    mel_len_wav = []
    dirs = os.listdir(path)
    for file in dirs:
        logmelspec,sr = get_spectrograms(path+file)
        mel_len_wav.append(len(logmelspec))
        mel_list.append(logmelspec.tolist())
    mel_numpy = tf.keras.preprocessing.sequence.pad_sequences(mel_list, maxlen=config.max_len, padding='post', dtype='float32')
    #print(len(mel_numpy[1000]))
    inputs = tf.convert_to_tensor(mel_numpy)
    return inputs,mel_len_wav

#用于训练stop_token
def tar_stop_token(mel_len_wav, mel_gts, max_len):
    tar_token = np.zeros((mel_gts.shape[0], max_len))
    for i in range(len(mel_len_wav)):
        j = mel_len_wav[i]
        tar_token[i, (j-1):] = 1
    return tar_token

#create_dataset
def create_dataset(batch_size, input_ids, mel_gts, tar_token):
    BUFFER_SIZE = len(input_ids)
    steps_per_epoch = BUFFER_SIZE // batch_size
    #dataset = tf.data.Dataset.from_tensor_slices((input_ids, mel_gts)).shuffle(BUFFER_SIZE)
    dataset = tf.data.Dataset.from_tensor_slices((input_ids, mel_gts, tar_token))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset, steps_per_epoch

if __name__ == '__main__':
    pass