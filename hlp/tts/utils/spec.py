import copy

import librosa
import numpy as np
import scipy
import tensorflow as tf


def get_spectrograms(audio_path: str, pre_emphasis: float, n_fft: int, n_mels: int,
                     hop_length: int, win_length: int, max_db: int, ref_db: int, top_db: int):
    """
    处理音频文件，转换成梅尔频谱和线性谱
    :param audio_path: 音频路径
    :param pre_emphasis: 预加重
    :param n_fft: FFT窗口大小
    :param n_mels: 产生的梅尔带数
    :param hop_length: 帧移
    :param win_length: 每一帧音频都由window()加窗，窗长win_length，然后用零填充以匹配N_FFT
    :param max_db: 峰值分贝值
    :param ref_db: 参考分贝值
    :param top_db: 峰值以下的阈值分贝值
    :return: 返回归一化后的梅尔频谱和线性谱，形状分别为(T, n_mels)和(T, 1+n_fft//2)
    """
    y, sr = librosa.load(audio_path, sr=None)
    y, _ = librosa.effects.trim(y, top_db=top_db)
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    # 短时傅里叶变换
    linear = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # 幅度谱
    mag = np.abs(linear)  # (1+n_fft//2, T)
    # mel频谱
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
    return mel, mag


def melspectrogram2wav(mel, max_db, ref_db, sr, n_fft, n_mels, preemphasis, n_iter, hop_length, win_length):
    """
    从线性幅度谱图生成wav文件
    :param mel: 梅尔谱
    :param sr: 采样率
    :param preemphasis: 预加重
    :param n_fft: FFT窗口大小
    :param n_mels: 产生的梅尔带数
    :param hop_length: 帧移
    :param win_length: 每一帧音频都由window()加窗，窗长win_length，然后用零填充以匹配N_FFT
    :param max_db: 峰值分贝值
    :param ref_db: 参考分贝值
    :param n_iter: 迭代指针
    """
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
    """
    已知幅度谱，未知相位谱，通过迭代生成相位谱，并用已
    知的幅度谱和计算得出的相位谱，重建语音波形的方法
    :param spectrogram: 幅度谱
    :param n_iter: 迭代指针
    :param n_fft: FFT窗口大小
    :param hop_length: 帧移
    :param win_length: 窗长win_length
    :return:
    """
    x_best = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        x_t = invert_spectrogram(x_best, hop_length, win_length)
        est = librosa.stft(x_t, n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        x_best = spectrogram * phase
    x_t = invert_spectrogram(x_best, hop_length, win_length)
    y = np.real(x_t)
    return y


def invert_spectrogram(spectrogram, hop_length, win_length):
    """
    spectrogram: [f, t]
    :param spectrogram: 幅度谱
    :param hop_length: 帧移
    :param win_length: 窗长win_length
    """
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")


def spec_distance(mel1, mel2):
    """
    计算mel谱之间的欧式距离
    :param mel1: 预测mel
    :param mel2: ground-true mel
    :return 两者之间的欧氏距离
    """
    mel1 = tf.transpose(mel1, [0, 2, 1])
    score = np.sqrt(np.sum((mel1 - mel2) ** 2))
    return score