import os
import re
import tensorflow as tf
import librosa
import numpy as np
from config2 import Tacotron2Config
#文字处理

def preprocess_sentence(s):
    s = s.lower().strip()
    s = re.sub(r"([?.!,])", r" \1 ", s)  # 切分断句的标点符号
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格
    s = re.sub(r"[^a-zA-Z?.!,]+", " ", s)
    s = s.strip()
    s = '<start> ' + s + ' <end>'
    return s

def process_text(text_data_path):
    sentences_list = []
    with open(text_data_path,"r") as f:
        sen_list = f.readlines()
    for sentence in sen_list[:]:
        sentences_list.append(preprocess_sentence(sentence.strip().lower()))
    return sentences_list

def tokenize(texts):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')  # 无过滤字符
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列
    print(sequences[-1])
    sequences_length = []
    for seq in sequences:
        sequences_length.append([len(seq)])
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,padding='post')

    return sequences,tokenizer

def Dataset_txt(path_to_file):
    en = process_text(path_to_file)
    en_seqs, en_tokenizer = tokenize(en)
    return en_seqs,en_tokenizer


#mel频谱处理
def get_spectrograms(fpath):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
 '''
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
    # Loading sound file
    y, sr = librosa.load(fpath, sr=None)

    # Trimming
    y, _ = librosa.effects.trim(y, top_db=top_db)

    # Preemphasis
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag

def Dataset_wave(path):
    mel_list = []
    dirs = os.listdir(path)
    for file in dirs:
        logmelspec,sr= get_spectrograms(path+file)
        mel_list.append(logmelspec.tolist())
    mel_numpy = tf.keras.preprocessing.sequence.pad_sequences(mel_list,padding='post',dtype='float32')
    #print(len(mel_numpy[1000]))
    inputs = tf.convert_to_tensor(mel_numpy)
    return inputs

#create_dataset
def create_dataset(batch_size,input_ids,mel_gts):
    BUFFER_SIZE = len(input_ids)
    steps_per_epoch = BUFFER_SIZE // batch_size
    dataset = tf.data.Dataset.from_tensor_slices((input_ids, mel_gts)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset,steps_per_epoch

if __name__ == '__main__':
    pass