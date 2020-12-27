import os
import sys

import librosa

sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("\\hlp\\")])
from hlp.tts.utils.spec import get_spectrograms

import numpy as np
import tensorflow as tf


# 处理语音文件
def load_wav(path, sample_rate):
    y = librosa.load(path, sr=sample_rate)[0]
    return y


def process_wav(path, sample_rate, peak_norm, voc_mode, bits, mu_law, preemphasis, n_fft, n_mels, hop_length, win_length
                , max_db, ref_db, top_db):
    y = load_wav(path, sample_rate)
    peak = np.abs(y).max()
    if peak_norm or peak > 1.0:
        y /= peak

    mel, _ = get_spectrograms(path, preemphasis, n_fft, n_mels, hop_length, win_length, max_db, ref_db, top_db)
    mel = tf.transpose(mel, (1, 0)).numpy()
    # mel = melspectrogram(y)
    if voc_mode == 'RAW':
        quant = encode_mu_law(y, mu=2 ** bits) if mu_law else float_2_label(y, bits=bits)
    elif voc_mode == 'MOL':
        quant = float_2_label(y, bits=16)

    return mel.astype(np.float32), quant.astype(np.int64)


def read_data(path, sample_rate, peak_norm, voc_mode, bits, mu_law, wav_name_list2, preemphasis, n_fft, n_mels,
              hop_length, win_length, max_db, ref_db, top_db):
    mel_list = []
    sig_list = []
    for file in wav_name_list2:
        m, x = process_wav(path + file + '.wav', sample_rate, peak_norm, voc_mode, bits, mu_law, preemphasis, n_fft,
                           n_mels, hop_length, win_length, max_db, ref_db, top_db)

        mel_list.append(m)
        sig_list.append(x)

    return mel_list, sig_list


def encode_mu_law(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2 ** bits - 1) / 2
    return x.clip(0, 2 ** bits - 1)


# 提取语音文件名
def process_wav_name(wav_path):
    datanames = os.listdir(wav_path)
    wav_name_list = []
    for i in datanames:
        wav_name_list.append(i[:10])
    return wav_name_list


def label_2_float(x, bits):
    return 2 * x / (2 ** bits - 1.) - 1.
