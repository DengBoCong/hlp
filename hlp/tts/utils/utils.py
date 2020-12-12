import librosa
import numpy as np


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
