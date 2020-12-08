'''
Author: PengKang6
Description: 语音特征提取方法
'''
import numpy as np
import soundfile as sf
from python_speech_features import mfcc, logfbank, delta


def wav_to_feature(wav_path, audio_feature_type):
    sig, sr = sf.read(wav_path)

    if audio_feature_type == "mfcc":
        return _get_mfcc_feature(sig, sr)
    elif audio_feature_type == "fbank":
        return _get_fbank_feature(sig, sr)


def _get_mfcc_feature(wavsignal, fs):
    # 输入为语音文件数学表示和采样频率，输出为语音的MFCC特征(默认13维)+一阶差分+二阶差分；
    feat_mfcc = mfcc(wavsignal, fs)
    feat_mfcc_d = delta(feat_mfcc, 2)
    feat_mfcc_dd = delta(feat_mfcc_d, 2)

    # (timestep, 39)
    wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
    return wav_feature.astype(np.float32)


def _get_fbank_feature(wavsignal, fs):
    # 输入为语音文件数学表示和采样频率，输出为语音的FBANK特征
    feat_fbank = logfbank(wavsignal, fs, nfilt=80)

    return feat_fbank.astype(np.float32)
