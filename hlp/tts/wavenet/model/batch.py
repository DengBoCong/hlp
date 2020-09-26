# -*- coding:utf-8 -*-
# @Time: 2020/9/25 17:20
# @Author:XuMengting
# @Email:1871489154@qq.com
import os
import numpy as np
from tqdm import tqdm
from hlp.tts.wavenet.model.audio import process_wav


# 加载音频处理好的sequence序列，二维（38，），每个值是采样点的值（0-255）之间用来表示音频特征
def load_set(desired_sample_rate, set_dirname, use_ulaw):
    ulaw_str = '_ulaw' if use_ulaw else ''
    cache_fn = os.path.join(set_dirname, 'processed_%d%s.npy' % (desired_sample_rate, ulaw_str))
    if os.path.isfile(cache_fn):
        full_sequences = np.load(cache_fn, allow_pickle=True)
    else:
        file_names = [fn for fn in os.listdir(set_dirname) if fn.endswith('.wav')]
        full_sequences = []
        for fn in tqdm(file_names):
            sequence = process_wav(desired_sample_rate, os.path.join(set_dirname, fn), use_ulaw)
            full_sequences.append(sequence)
        np.save(cache_fn, full_sequences)
    return full_sequences


# 产生输入和标签对
def random_batch_generator(fragment_length, batch_size, desired_sample_rate, set_dirname, use_ulaw):
    full_sequences = load_set(desired_sample_rate, set_dirname, use_ulaw)
    # full_sequence里面每一个小的seq的长度也就是采样点的个数的列表[576231,379700,598004,337306,602036,367488,615053]
    lengths = [x.shape[0] for x in full_sequences]
    # 小的seq的数量例如（38，）这个len(full_sequences)=38
    nb_sequences = len(full_sequences)
    while True:
        sequence_indices = np.random.randint(0, nb_sequences, batch_size)
        # 生成的是0到38之间比如batch_size=16则生成一个array[17 36 19 25  7 29 23 26 18 23 31 21 37  9 11 37]他们是小的seq序号索引号
        batch_inputs = []
        batch_outputs = []
        for i, seq_i in enumerate(sequence_indices):
            l = lengths[seq_i]
            """
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
            # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
            offset = np.squeeze(np.random.randint(0, l - fragment_length, 1))
            # 在0到l-16000之间任意找一个切口点
            batch_inputs.append(full_sequences[seq_i][offset:offset + fragment_length])
            batch_outputs.append(full_sequences[seq_i][offset + 1:offset + fragment_length + 1])

       # 其中有16个array, 其中每个array的[]里面有16000个数
       # batch_inputs = [array([30, 36, 44, ..., 75, 68, 70], dtype=uint8),array([169, 175, 176, ..., 75, 73, 69], dtype=uint8)]
       # batch_outputs = [array([36, 44, 165, ..., 68, 70, 95], dtype=uint8), array([175, 176, 171, ..., 73, 69, 72], dtype=uint8)]
        return batch_inputs, batch_outputs


full_seq = load_set(160124,r'E:\git\hlp\hlp\tts\wavenet\data\train',True)
print(full_seq)
print(full_seq[0].shape)
# 测试fragment长度，batch_size大小采样率,这里的batch_size是指从总的数据中截取多少出来作为总的训练数据，所以是总的训练数据
batch_inputs, batch_outputs = random_batch_generator(6000, 10, 160124, r'E:\git\hlp\hlp\tts\wavenet\data\train', True)
print(batch_inputs)
print(batch_outputs)
print(batch_inputs[0].shape)
