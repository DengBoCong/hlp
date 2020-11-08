import os
import re
import tensorflow as tf
import librosa
import numpy as np
from config2 import Tacotron2Config
import io
import json
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

#提取语音文件名
def process_wav_name(wav_path):
    datanames = os.listdir(wav_path)
    wav_name_list = []
    for i in datanames:
        wav_name_list.append(i[:10])
    return wav_name_list

#用语音文件名映射保存文本
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
def process_text(sentence_list):
    en_sentences = [preprocess_sentence(s) for s in sentence_list]
    return en_sentences

def tokenize(texts, save_path, name):
    if name == "train":
        #准备train之前要保存字典
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='UNK')  # 无过滤字符
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列
        json_string = tokenizer.to_json()
        with open(save_path, 'w') as f:
            json.dump(json_string, f)
        sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
        vocab_size = len(tokenizer.word_index)+1
        return sequences, vocab_size
    else:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')  # 无过滤字符
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列
        # print(sequences[-1])
        sequences_length = []
        for seq in sequences:
            sequences_length.append([len(seq)])
        sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
        vocab_size = len(tokenizer.word_index) + 1
        return sequences, vocab_size

#提取字典
def _get_tokenizer_keras(path):
    """从指定路径加载保存好的字典"""
    with open(path) as f:
        json_string = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
    vocab_size = len(tokenizer.word_index)
    return tokenizer, vocab_size

#训练的时候对取得的句子列表处理
def dataset_txt(sentence_list, save_path, name):
    en = process_text(sentence_list)
    en_seqs, vocab_size = tokenize(en, save_path, name)
    return en_seqs, vocab_size

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
    return inputs, mel_len_wav

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