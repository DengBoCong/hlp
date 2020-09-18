# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:20:05 2020

@author: 九童
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import dct

def MFCC(sample_rate, signal):    
# 原始数据,读取前3.5s 的数据    original_signal = signal[0:int(3.5*sample_rate)]
    original_signal = signal[0:int(10*sample_rate)]

    signal_num = np.arange(len(signal))
    sample_num = np.arange(len(original_signal))
    '''
    # 绘图 01
    plt.figure(figsize=(11,7), dpi=500)
    
    plt.subplot(211)
    plt.plot(signal_num/sample_rate, signal, color='black')
    plt.plot(sample_num/sample_rate, original_signal, color='blue')
    plt.ylabel("Amplitude")
    plt.title("signal of Voice")
    
    plt.subplot(212)
    plt.plot(sample_num/sample_rate, original_signal, color='blue')
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude") 
    plt.title("3.5s signal of Voice ")
    
    plt.savefig('mfcc_01.png')
    '''
    
    pre_emphasis = 0.97
    emphasized_signal = np.append(original_signal[0], original_signal[1:] - pre_emphasis * original_signal[:-1])
    emphasized_signal_num = np.arange(len(emphasized_signal))
    
    '''
    # 绘图 02
    plt.figure(figsize=(11,4), dpi=500)
    
    plt.plot(emphasized_signal_num/sample_rate, emphasized_signal, color='blue')
    plt.xlabel("Time (sec)", fontsize=14)
    plt.ylabel("Amplitude", fontsize=14)
    plt.title("emphasized signal of Voice", fontsize=14)
    
    plt.savefig('mfcc_02.png')
    '''
    
    # 分帧
    frame_size = 0.025
    frame_stride = 0.1
    frame_length = int(round(frame_size*sample_rate))
    frame_step = int(round(frame_stride*sample_rate)) 
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length-frame_length))/frame_step))
    
    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.append(emphasized_signal, np.zeros((pad_signal_length - signal_length)))
    
    indices = np.tile(np.arange(0,frame_length),(num_frames,1))+np.tile(np.arange(0,num_frames*frame_step,frame_step), (frame_length, 1)).T
    frames = pad_signal[np.mat(indices).astype(np.int32, copy=False)]
    '''
    # 绘图 03
    plt.figure(figsize=(11,4), dpi=500)
    
    pad_signal_num = np.arange(len(pad_signal))
    plt.plot(pad_signal_num/sample_rate, pad_signal, color='blue')
    plt.xlabel("Time (sec)", fontsize=14)
    plt.ylabel("Amplitude", fontsize=14)
    plt.title("pad signal of Voice", fontsize=14)
    
    plt.savefig('mfcc_03.png')
    '''
    
    # 汉明窗
    N = 200
    x = np.arange(N)
    y = 0.54 * np.ones(N) - 0.46 * np.cos(2*np.pi*x/(N-1))
    
    '''
    plt.plot(x, y, label='Hamming')
    plt.xlabel("Samples")
    plt.ylabel("Amplitude") 
    plt.legend()
    plt.savefig('hamming.png', dpi=500)
    '''
    
    # 加汉明窗
    frames *= np.hamming(frame_length)
    # Explicit Implementation
    # frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))
    
    # 傅里叶变换和功率谱
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)
    
    # 将频率转换为Mel频率
    low_freq_mel = 0
    
    nfilt = 40
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    
    num_ceps = 256
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape
    
    mfcc
    
    num_ceps = 256
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape
    
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    
    '''
    # 绘图 04
    plt.figure(figsize=(11,7), dpi=500)
    
    plt.subplot(211)
    plt.imshow(np.flipud(filter_banks.T), cmap=plt.cm.jet, 
               aspect=0.2, extent=[0,filter_banks.shape[1],0,
                                   filter_banks.shape[0]]) #画热力图
    plt.title("MFCC")
    
    plt.subplot(212)
    plt.imshow(np.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.2, 
               extent=[0,mfcc.shape[0],0,mfcc.shape[1]])#热力图
    plt.title("MFCC")
    
    plt.savefig('mfcc_04.png')
    '''
    return mfcc


