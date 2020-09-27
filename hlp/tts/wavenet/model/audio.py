# -*- coding:utf-8 -*-
# @Time: 2020/9/24 14:36
# @Author:XuMengting
# @Email:1871489154@qq.com
# 主要写音频文件的初处理
import warnings
import numpy as np
import scipy.io.wavfile
import scipy.signal
# 测试用
import matplotlib.pyplot as plt


# 数据处理：波形振幅-->one-hot
# 读取音频文件名获得采样率和音频振幅-->单声道处理-->转为浮点数(0~1)-->转化为mulaw-->resample到目标采样率--->转化为uint8(0~(2^8-1=255))数据
def process_wav(desired_sample_rate, filename, use_ulaw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
    file_sample_rate, audio = scipy.io.wavfile.read(filename)
    audio = ensure_mono(audio)
    audio = wav_to_float(audio)
    if use_ulaw:
        audio = ulaw(audio)
    audio = ensure_sample_rate(desired_sample_rate, file_sample_rate, audio)
    audio = float_to_uint8(audio)
    return audio


# 确保音频是单声道
def ensure_mono(raw_audio):
    if raw_audio.ndim == 2:
        raw_audio = raw_audio[:, 0]
    return raw_audio


# 振幅变成浮点数（0~1）之间
def wav_to_float(x):
    try:
        max_value = np.iinfo(x.dtype).max
        min_value = np.iinfo(x.dtype).min
    except:
        max_value = np.finfo(x.dtype).max
        min_value = np.iinfo(x.dtype).min
    x = x.astype('float64', casting='safe')
    x -= min_value
    x /= ((max_value - min_value) / 2.)
    x -= 1.
    return x


# 浮点数float（0~1）转为（0~255）uint8数据
def float_to_uint8(x):
    x += 1.
    x /= 2.
    uint8_max_value = np.iinfo('uint8').max
    x *= uint8_max_value
    x = x.astype('uint8')
    return x


# mu律转化
def ulaw(x, u=255):
    x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
    return x


# 将目标文件转化为目标采样率
def ensure_sample_rate (desired_sample_rate, file_sample_rate, mono_audio):
    if file_sample_rate != desired_sample_rate:
        mono_audio = scipy.signal.resample_poly(mono_audio, desired_sample_rate, file_sample_rate)
    return mono_audio


def one_hot(x):
    return np.eye(256, dtype='uint8')[x.astype('uint8')]


"""
# 测试
audio = process_wav(160124, r"E:\git\hlp\hlp\tts\wavenet\data\train\Taylor_Swift_-_Blank_Space.wav", True)
print(audio.shape)
plt.xlabel("numbers of sample")
plt.ylabel("value of sample")
plt.plot(audio)
plt.show()
"""