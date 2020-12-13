import copy
import scipy
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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
    """
    spectrogram: [f, t]
    """
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")


def spec_distance(mel1, mel2):
    """
    计算mel谱之间的欧式距离
    """
    mel1 = tf.transpose(mel1, [0, 2, 1])
    score = np.sqrt(np.sum((mel1 - mel2) ** 2))
    return score


def plot_spectrogram_to_numpy(spectrogram):
    """
    下面两个方法没使用，暂时保留
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    fig.canvas.draw()
    data = _save_figure_to_numpy(fig)
    plt.close()
    return data


def _save_figure_to_numpy(fig):
    # 保存成numpy
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def get_phoneme_dict_symbols(unknown: str = "<unk>", eos: str = "~"):
    """
    用于创建音素文件，方便在pre_treat中使用
    :param unknown: 未登录词
    :param eos: 结尾词
    :return: 字典和39个原始音素和字符的集合
    """
    symbols = [
        'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
        'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
        'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
        'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
        'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
        'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
        'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
    ]

    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '
    phonemes = ['@' + s for s in symbols]
    symbols_list = [unknown, eos] + list(chars) + phonemes

    dict_set = {s: i for i, s in enumerate(symbols_list)}

    return dict_set, set(symbols)
