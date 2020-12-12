import copy

import librosa
import numpy as np
import scipy
import tensorflow as tf



def melspectrogram2wav(mel, max_db, ref_db, sr, n_fft, n_mels, preemphasis, n_iter, hop_length, win_length):
    mel = (np.clip(mel, 0, 1) * max_db) - max_db + ref_db
    # 转为幅度谱
    mel = np.power(10.0, mel * 0.05)
    m = _mel_to_linear_matrix(sr, n_fft, n_mels)
    mag = np.dot(m, mel)
    # 波形重构
    wav = griffin_lim(mag, n_iter, n_fft, hop_length, win_length)
    wav = scipy.signal.lfilter([1], [1, -preemphasis], wav)
    # 剪裁
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)


def _mel_to_linear_matrix(sr, n_fft, n_mels):
    m = librosa.filters.mel(sr, n_fft, n_mels)
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))


def griffin_lim(spectrogram, n_iter, n_fft, hop_length, win_length):
    X_best = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        X_t = invert_spectrogram(X_best, hop_length, win_length)
        est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best, hop_length, win_length)
    y = np.real(X_t)
    return y


def invert_spectrogram(spectrogram, hop_length, win_length):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")


# 计算mel谱之间的欧式距离
def spec_distance(mel1, mel2):
    mel1 = tf.transpose(mel1, [0, 2, 1])
    score = np.sqrt(np.sum((mel1 - mel2) ** 2))
    return score