# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 08:23:53 2020

Spec augmentation 计算函数

    'SpecAugment'有3个步骤为音频数据增强
    第一步是使用Tensorflow的image_sparse_warp函数进行时间扭曲
    第二步是频率掩蔽
    最后一步是时间掩蔽。

@author: 九童
"""
import librosa
import librosa.display
import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp
import numpy as np
import matplotlib.pyplot as plt
import argparse


def sparse_warp(mel_spectrogram, time_warping_para=80):
    print("时间扭曲")

    """
    # 参数:
      mel_spectrogram(numpy array): 你想要扭曲和屏蔽的音频文件路径.
      time_warping_para(float): 增强参数, "时间扭曲参数 W".
       如果为none, 对于LibriSpeech数据集默认为80.

    # Returns
      mel_spectrogram(numpy array): 扭曲和掩蔽后的梅尔频谱图.
      τ个时间步的log mel 频谱图
      (W,τ-W)范围内的随机点，向左或向右平移w距离，w从(0，W）的均匀分布中挑出。
    边界上有六个固定点，W是time warp parameter

    """

    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]
    print("n: {}".format(n))
    print("v: {}".format(v))
    print("time_warping_para: {}".format(time_warping_para))
    """
    n: 256
    v: 92
    time_warping_para: 80
    pt: 105
    """
    # n为该频谱图的时间步
    # v为该频谱图的频率
    # 步骤1 : 时间扭曲
    # 图像扭曲控制点设置。
    # 源
    pt = tf.random.uniform([], time_warping_para, n - time_warping_para, tf.int32)  # radnom point along the time axis
    print("pt: {}".format(pt))  # (80,176)之间的一个随机数
    src_ctr_pt_freq = tf.range(v // 2)  # [0,46)的一系列数字control points on freq-axis
    src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt  # 返回一个全为1，形状相同的张量，control points on time-axis
    # src_ctr_pts = tf.stack((src_ctr_pt_time, src_ctr_pt_freq), -1)
    src_ctr_pts = tf.stack((src_ctr_pt_freq, src_ctr_pt_time), -1)
    src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)
    # 目标
    w = tf.random.uniform([], 0, time_warping_para, tf.int32)  # distance
    # ？？？
    # w = tf.random.uniform([], -time_warping_para, time_warping_para, tf.int32)  # distance
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    # dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq), -1)
    dest_ctr_pts = tf.stack((dest_ctr_pt_freq, dest_ctr_pt_time), -1)
    dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)
    # 扭曲
    source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
    dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0)  # (1, v//2, 2)

    warped_image, _ = sparse_image_warp(mel_spectrogram,
                                        source_control_point_locations,
                                        dest_control_point_locations)
    return warped_image


def frequency_masking(mel_spectrogram, v, frequency_masking_para=27, frequency_mask_num=2):
    print("频率掩蔽法")

    """
    频率掩蔽法
    # 参数:
      mel_spectrogram(numpy array): 你想要扭曲和屏蔽的音频文件路径.
      frequency_masking_para(float): 增强参数, "频率掩蔽参数 F"
        如果为none, 对于LibriSpeech数据集默认为27
      frequency_mask_num(float): 频率掩蔽线的数目, "m_F".
        如果为none, 对于LibriSpeech数据集默认为1

    # Returns
      mel_spectrogram(numpy array): 扭曲和掩蔽后的梅尔频谱图.
      沿频域轴方向的[f0,f0+f)范围内的连续频率通道进行掩蔽，其中0=<f0<ν，
      f服从[0,F]均匀分布，f0从[0,v-f)中选出，v是mel频率通道数，F是频率掩蔽参数
    """
    # 步骤 2 : 频率掩蔽
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    for i in range(frequency_mask_num):
        f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
        v = tf.cast(v, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=v - f, dtype=tf.int32)
        # 错
        # warped_mel_spectrogram[f0:f0 + f, :] = 0 mel_spectrogram.shape: (1, 256, 92, 1)
        mask = tf.concat((tf.ones(shape=(1, n, v - f0 - f, 1)),
                          tf.zeros(shape=(1, n, f, 1)),
                          tf.ones(shape=(1, n, f0, 1)),
                          ), 2)
        mel_spectrogram = mel_spectrogram * mask
    return tf.cast(mel_spectrogram, dtype=tf.float32)


def time_masking(mel_spectrogram, tau, time_masking_para=100, time_mask_num=1):
    print("时间掩蔽法")
    """
    时间掩蔽法
    # 参数:
      mel_spectrogram(numpy array): 你想要扭曲和屏蔽的音频文件路径.
      time_masking_para(float): 增强参数, "时间掩蔽参数 T"
        如果为none, 对于LibriSpeech数据集默认为100
      time_mask_num(float): 时间掩蔽线的数目, "m_T".
        如果为none, 对于LibriSpeech数据集默认为1

    # Returns
      mel_spectrogram(numpy array): 扭曲和掩蔽后的梅尔频谱图.τ是时间片长度
      沿时间轴方向的[t0,t0+t)范围内的连续时间步进行掩蔽，其中0=<t0<τ,t服从[0,T]均匀分布，T是时间掩蔽参数

    """
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    # Step 3 : Time masking
    for i in range(time_mask_num):
        t = tf.random.uniform([], minval=0, maxval=time_masking_para, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=tau - t, dtype=tf.int32)

        # mel_spectrogram[:, t0:t0 + t] = 0 ，mel_spectrogram.shape: (1, 256, 92, 1)
        mask = tf.concat((tf.ones(shape=(1, n - t0 - t, v, 1)),
                          tf.zeros(shape=(1, t, v, 1)),
                          tf.ones(shape=(1, t0, v, 1)),
                          ), 1)
        mel_spectrogram = mel_spectrogram * mask
    return tf.cast(mel_spectrogram, dtype=tf.float32)


def spec_augment(mel_spectrogram):
    v = mel_spectrogram.shape[0]
    tau = mel_spectrogram.shape[1]

    warped_mel_spectrogram = sparse_warp(mel_spectrogram)

    warped_frequency_spectrogram = frequency_masking(mel_spectrogram, v=v)

    warped_time_sepctrogram = time_masking(mel_spectrogram, tau=tau)

    return warped_mel_spectrogram


def plot_spectrogram(mel_spectrogram, title):
    """可视化SpecAugment的第一个结果
    # 参数:
      mel_spectrogram(ndarray): 梅尔频谱可视化
      title(String): 梅尔频谱图名
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :, 0], ref=np.max), y_axis='mel', fmax=8000,
                             x_axis='time')
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spec Augment')
    parser.add_argument('--audio-path', default='./data/number/wav_test/0_jackson_1.wav',
                        help='The audio file.')
    parser.add_argument('--time-warp-para', default=80,
                        help='time warp parameter W')
    parser.add_argument('--frequency-mask-para', default=100,
                        help='frequency mask parameter F')
    parser.add_argument('--time-mask-para', default=27,
                        help='time mask parameter T')
    parser.add_argument('--masking-line-number', default=1,
                        help='masking line number')

    args = parser.parse_args()
    audio_path = args.audio_path
    time_warping_para = args.time_warp_para
    time_masking_para = args.frequency_mask_para
    frequency_masking_para = args.time_mask_para
    masking_line_number = args.masking_line_number
    print(audio_path)
    audio, sampling_rate = librosa.load(audio_path)

    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=sampling_rate,
                                                     n_mels=256,
                                                     hop_length=128,
                                                     fmax=8000)

    # reshape spectrogram shape to [batch_size, time, frequency, 1]
    shape = mel_spectrogram.shape

    mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1], 1))

    # Show Raw mel-spectrogram
    plot_spectrogram(mel_spectrogram=mel_spectrogram,
                     title="Raw Mel Spectrogram")

    # Show time warped & masked spectrogram
    plot_spectrogram(mel_spectrogram=spec_augment(mel_spectrogram),
                     title="tensorflow Warped & Masked Mel Spectrogram")
