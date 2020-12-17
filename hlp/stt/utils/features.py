import numpy as np
import soundfile as sf
from python_speech_features import mfcc, logfbank, delta


def wav_to_feature(wav_path, audio_feature_type):
    """提取语音文件语音特征

    :param wav_path: 音频文件路径
    :param audio_feature_type: 特征类型
    :return: shape为(timestep, dim)的音频特征
    """
    sig, sr = sf.read(wav_path)

    if audio_feature_type == "mfcc":
        return get_mfcc_(sig, sr)
    elif audio_feature_type == "fbank":
        return get_fbank(sig, sr)


def get_mfcc_(wavsignal, fs):
    # 输入为语音文件数学表示和采样频率，输出为语音的MFCC特征(默认13维)+一阶差分+二阶差分；
    feat_mfcc = mfcc(wavsignal, fs)
    feat_mfcc_d = delta(feat_mfcc, 2)
    feat_mfcc_dd = delta(feat_mfcc_d, 2)

    # (timestep, 39)
    wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
    return wav_feature.astype(np.float32)


def get_fbank(wavsignal, fs):
    # 输入为语音文件数学表示和采样频率，输出为语音的FBANK特征
    feat_fbank = logfbank(wavsignal, fs, nfilt=80)

    return feat_fbank.astype(np.float32)
