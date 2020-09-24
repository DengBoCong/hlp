#主要写数据的处理和分批处理
import warnings
import os
import librosa
import numpy as np
import scipy.io.wavfile
import matplotlib.plot as plt

#数据处理：波形振幅-->one-hot
#读取音频文件名获得采样率和音频振幅-->单声道处理-->转为浮点数(0~1)-->转化为mulaw
# -->resample到目标采样率--->转化为uint8(0~(2^8-1=255))数据
def process_wav(desired_sample_rate, filename, use_ulaw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
    channels = scipy.io.wavfile.read(filename)
    file_sample_rate , audio = channels
    audio = ensure_mono(audio)
    audio = wav_to_float(audio)
    if use_ulaw:
        audio = ulaw(audio)
    audio = ensure_sample_rate(desired_sample_rate,file_sample_rate,audio)
    audio = float_to_uint8(audio)
    return audio
#确保音频是单声道
def ensure_mono(raw_audio):
    if raw_audio.ndim == 2:
        raw_audio = raw_audio[:, 0]
    return raw_audio
#振幅变成浮点数（0~1）之间
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
#浮点数float（0~1）转为（0~255）uint8数据
def float_to_uint8(x):
    x += 1.
    x /= 2.
    uint8_max_value = np.iinfo('uint8').max
    x *= uint8_max_value
    x = x.astype('uint8')
    return x
#mu律转化
def ulaw(x, u=255):
    x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
    return x
#将目标文件转化为目标采样率
def ensure_sample_rate(desired_sample_rate, file_sample_rate, mono_audio):
    if file_sample_rate != desired_sample_rate:
        mono_audio = scipy.signal.resample_poly(mono_audio, desired_sample_rate, file_sample_rate)
    return mono_audio
def one_hot(x):
    return np.eye(256, dtype='uint8')[x.astype('uint8')]


#加载音频处理好的sequence序列，二维（38，），每个值是采样点的值（0-255）之间用来表示音频特征
def load_set(desired_sample_rate, set_dirname, use_ulaw):
    ulaw_str = '_ulaw' if use_ulaw else ''
    cache_fn = os.path.join(set_dirname, 'processed_%d%s.npy' % (desired_sample_rate, ulaw_str))
    if os.path.isfile(cache_fn):
        full_sequences = np.load(cache_fn,allow_pickle=True)
    else:
        file_names = [fn for fn in os.listdir(set_dirname) if fn.endswith('.wav')]
        full_sequences = []
        for fn in tqdm(file_names):
            sequence = process_wav(desired_sample_rate, os.path.join(set_dirname, fn), use_ulaw)
            full_sequences.append(sequence)
        np.save(cache_fn, full_sequences)
    return full_sequences

#处理好训练数据以输入和期望输出的形式
def random_batch_generator( fragment_length, batch_size,desired_sample_rate, set_dirname, use_ulaw):
    full_sequences = load_set(desired_sample_rate, set_dirname, use_ulaw)
    #full_sequence里面每一个小的seq的长度也就是采样点的个数的列表[576231,379700,598004,337306,602036,367488,615053]
    lengths = [x.shape[0] for x in full_sequences]
    #小的seq的数量例如（38，）这个len(full_sequences)=38
    nb_sequences = len(full_sequences)
    while True:
        sequence_indices = np.random.randint(0, nb_sequences, batch_size)
        #生成的是0到38之间比如batch_size=16则生成一个array[17 36 19 25  7 29 23 26 18 23 31 21 37  9 11 37]他们是小的seq序号索引号
        batch_inputs = []
        batch_outputs = []
        for i, seq_i in enumerate(sequence_indices):
            l = lengths[seq_i]
            """
            576231
            604916
            593050
            405850
            544666
            601114
            309082
            559296
            604916
            375898
            329588
            387303
            637517
            668160
            321178
            405850
            """
            #从数组的形状中删除单维度条目，即把shape中为1的维度去掉
            offset = np.squeeze(np.random.randint(0, l - fragment_length, 1))
            #在0到l-16000之间任意找一个切口点
            batch_inputs.append(full_sequences[seq_i][offset:offset + fragment_length])
            batch_outputs.append(full_sequences[seq_i][offset + 1:offset + fragment_length + 1])
            """
            其中有16个array,其中每个array的[]里面有16000个数
        batch_inputs=[array([30, 36, 44, ..., 75, 68, 70], dtype=uint8), array([169, 175, 176, ...,  75,  73,  69], dtype=uint8)]
        batch_outputs=[array([ 36,44, 165, ...,  68,  70,  95], dtype=uint8), array([175, 176, 171, ...,  73,  69,  72], dtype=uint8)]
            """
        return batch_inputs,batch_outputs

