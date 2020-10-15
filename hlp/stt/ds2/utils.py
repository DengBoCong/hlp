# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:15:16 2020

@author: 彭康
"""

import os
import time
import wave
import librosa
import numpy as np
import pyaudio
import tensorflow as tf
import config


#基于数据文本规则的行获取
def text_process(str):
    """
    #文本每行"string\n"
    return str.strip().lower()
    """
    #当前数据文本的每行为'index string\n'
    return str.strip().split(" ",1)[1].lower()

#音频的处理
def wav_to_mfcc(n_mfcc,wav_path):
    #加载音频
    y, sr = librosa.load(wav_path,sr=None)
    #提取mfcc(返回list(timestep,n_mfcc))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).transpose(1,0).tolist()
    return mfcc

#字符序列list转成数字label list
def text_to_int_sequence(text,char_map):
    int_sequence = []
    for ch in text:
        if ch == ' ':
            i = char_map['<space>']
        else:
            i = char_map[ch]
        int_sequence.append(i)
    return int_sequence

#数字label list转成字符序列list
def int_to_text_sequence(seq,index_map):
    text_sequence = []
    for i in seq:
        if i>=1 and i<=(len(index_map)):
            ch = index_map[i]
        else:
            ch = ''
        text_sequence.append(ch)
    #"".join(text_sequence)就会转成字符串
    return text_sequence
        
#输入的两个参数均是字符串的list,是wer计算的入口
def wers(originals, results):
    count = len(originals)
    try:
        assert count > 0
    except:
        print(originals)
        raise("ERROR assert count>0 - looks like data is missing")
    rates = []
    mean = 0.0
    assert count == len(results)
    for i in range(count):
        rate = _wer(originals[i], results[i])
        mean = mean + rate
        rates.append(rate)
    return rates, mean / float(count)

def _wer(original, result):
    r"""
    The WER is defined as the editing/Levenshtein distance on word level
    divided by the amount of words in the original text.
    In case of the original having more words (N) than the result and both
    being totally different (all N words resulting in 1 edit operation each),
    the WER will always be 1 (N / N = 1).
    """
    # The WER ist calculated on word (and NOT on character) level.
    # Therefore we split the strings into words first:
    original = original.split()
    result = result.split()
    return _levenshtein(original, result) / float(len(original))

def lers(originals, results):
    count = len(originals)
    assert count > 0
    rates = []
    norm_rates = []

    mean = 0.0
    norm_mean = 0.0

    assert count == len(results)
    for i in range(count):
        rate = _levenshtein(originals[i], results[i])
        mean = mean + rate

        normrate = (float(rate) / len(originals[i]))

        norm_mean = norm_mean + normrate

        rates.append(rate)
        norm_rates.append(round(normrate, 4))

    return rates, (mean / float(count)), norm_rates, (norm_mean/float(count))

def _levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = list(range(n+1))
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

#获取麦克风录音并保存在filepath中
def record(file_path):
        CHUNK = 256
        FORMAT = pyaudio.paInt16
        CHANNELS = 1                # 声道数
        RATE = 16000               # 采样率
        RECORD_SECONDS = config.configs_record()["record_times"]        #录音时长
        WAVE_OUTPUT_FILENAME = file_path
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        print("开始录音：请在%d秒内输入语音:" % (RECORD_SECONDS))
        frames = []
        for i in range(1,int(RATE / CHUNK * RECORD_SECONDS)+1):
            data = stream.read(CHUNK)
            frames.append(data)
            if (i % (RATE / CHUNK)) == 0:
                print('\r%s%d%s' % ("剩余",int(RECORD_SECONDS-(i//(RATE / CHUNK))),"秒"),end="")
        print("\n录音结束\n")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

#从char_set.txt里边构建index_map和char_map
def get_index_and_char_map():
    char_set_path = config.configs_other()["char_set_path"]
    index_map = {}
    char_map = {}
    with open(char_set_path,"r") as f:
        map_list = f.readlines()
    for i in range(len(map_list)):
        ci = map_list[i].strip().split()
        char_map[ci[0]] = int(ci[1])
        if ci[0] == "<space>":
            index_map[int(ci[1])] = " "
        else:
            index_map[int(ci[1])] = ci[0]
    return index_map,char_map


if __name__ == "__main__":
    """
    #通过断言进行测试
    originals1 = ["a bcde fghij kl"]
    results1 = ["a bcde fgh ijk l"]
    originals2 = ["我是中国人民"]
    results2 = ["我美国人"]
    rates_wers,aver_wers=wers(originals1,results1)
    #1为增加的词，2为替换的词，0为删除的词，4为原始的词的数量
    assert rates_wers[0] == (1+2+0)/4
    rates_lers,aver_lers,norm_rates_lers,norm_aver_lers=lers(originals2,results2)
    #0为增加的字符，1为替换的字符，2为删除的字符，6为原始的字符数量
    assert rates_lers[0] == (0+1+2)
    assert norm_rates_lers[0] == (0+1+2)/6
    """
    pass
