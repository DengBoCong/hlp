import numpy as np
import soundfile as sf
import tensorflow as tf
from python_speech_features import mfcc, logfbank, delta


def wav_to_feature(wav_path, audio_feature_type):
    """
    提取语音文件语音特征
    :param wav_path: 音频文件路径
    :param audio_feature_type: 特征类型
    :return: shape为(timestep, dim)的音频特征
    """
    sig, sr = sf.read(wav_path)

    if audio_feature_type == "mfcc":
        return get_mfcc_(sig, sr)
    elif audio_feature_type == "fbank":
        return get_fbank(sig, sr)


def get_mfcc_(wav_signal, sr):
    """
    :param wav_signal: 音频数字信号
    :param sr: 采样率
    输入为语音文件数学表示和采样频率，输出为语音的MFCC特征(默认13维)+一阶差分+二阶差分；
    """
    feat_mfcc = mfcc(wav_signal, sr)
    feat_mfcc_d = delta(feat_mfcc, 2)
    feat_mfcc_dd = delta(feat_mfcc_d, 2)

    # (timestep, 39)
    wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
    return wav_feature.astype(np.float32)


def get_fbank(wav_signal, sr):
    """
    :param wav_signal: 音频数字信号
    :param sr: 采样率
    输入为语音文件数学表示和采样频率，输出为语音的FBANK特征
    """
    feat_fbank = logfbank(wav_signal, sr, nfilt=80)

    return feat_fbank.astype(np.float32)


def get_input_and_length(audio_path_list, audio_feature_type, max_len):
    """
    获得语音文件的特征和长度
    :param audio_path_list: 语音文件列表
    :param audio_feature_type: 语音特征类型
    :param max_len: 最大补齐长度
    :return: 补齐后的语音特征数组，每个语音文件的帧数
    """
    audio_feature_list = []
    input_length_list = []
    for audio_path in audio_path_list:
        audio_feature = wav_to_feature(audio_path, audio_feature_type)
        audio_feature_list.append(audio_feature)
        input_length_list.append([audio_feature.shape[0]])

    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(audio_feature_list,
                                                                 maxlen=max_len,
                                                                 dtype='float32',
                                                                 padding='post'
                                                                 )
    input_length = tf.convert_to_tensor(input_length_list)

    return input_tensor, input_length


def max_audio_length(audio_path_list, audio_feature_type):
    """ 获得语音特征帧最大长度

    注意：这个方法会读取所有语音文件，并提取特征.

    :param audio_path_list: 语音文件列表
    :param audio_feature_type: 语音特征类型
    :return: 最大帧数
    """
    return max(wav_to_feature(audio_path, audio_feature_type).shape[0] for audio_path in audio_path_list)


if __name__ == "__main__":
    pass
