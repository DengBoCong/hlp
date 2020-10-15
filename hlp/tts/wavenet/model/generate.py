# -*- coding:utf-8 -*-
# @Time: 2020/9/26 17:49
# @Author:XuMengting
# @Email:1871489154@qq.com
import os
import random
import wave
import numpy as np
from picklable_itertools import xrange
from tqdm import tqdm
from hlp.tts.wavenet.model.model import model
from hlp.tts.wavenet.model.audio import process_wav, one_hot

predict_seconds = 1
sample_argmax = False
sample_temperature = 1.0  # Temperature for sampling. > 1.0 for more exploring, < 1.0 for conservative samples.
predict_use_softmax_as_input = False  # Uses the softmax rather than the argmax as in input for the next step.
predict_initial_input = None


# 创建一个后缀为.wav文件名用于写入声音，返回一个生成的sample的名称
def make_sample_name(epoch, predict_seconds, predict_use_softmax_as_input, sample_argmax, sample_temperature, seed):
    sample_str = ''
    if predict_use_softmax_as_input:
        sample_str += '_soft-in'
    if sample_argmax:
        sample_str += '_argmax'
    else:
        sample_str += '_sample'
        if sample_temperature:
            sample_str += '-temp-%s' % sample_temperature
    sample_name = 'sample_epoch-%05d_%02ds_%s_seed-%d.wav' % (epoch, int(predict_seconds), sample_str, seed)
    return sample_name


# 打开sample_filename.wav文件，配置声道数，量化位数，采样频率头部
def make_sample_stream(desired_sample_rate, sample_filename):
    sample_file = wave.open(sample_filename, mode='w')
    sample_file.setnchannels(1)
    sample_file.setframerate(desired_sample_rate)
    sample_file.setsampwidth(1)
    return sample_file


# write_samples(sample_stream, [output_val])在配置好头部的wav文件中写入声音流的值
def write_samples(sample_file, out_val, use_ulaw):
    s = np.argmax(out_val, axis=-1).astype('uint8')
    if use_ulaw:
        s = ulaw2lin(s)
    s = bytearray(list(s))
    sample_file.writeframes(s)


# 将输出的分布distribution变为每一个采样点的value
def draw_sample(output_dist, sample_temperature, sample_argmax, _rnd):
    if sample_argmax:
        output_dist = np.eye(256)[np.argmax(output_dist, axis=-1)]
    else:
        if sample_temperature is not None:
            output_dist = softmax(output_dist, sample_temperature)
        output_dist = output_dist / np.sum(output_dist + 1e-7)
        output_dist = random.multinomial(1, output_dist)
    return output_dist


def softmax(x, temp, mod=np):
    x = mod.log(x) / temp
    e_x = mod.exp(x - mod.max(x, axis=-1))
    return e_x / mod.sum(e_x, axis=-1)


def ulaw2lin(x, u=255.):
    max_value = np.iinfo('uint8').max
    # max_value = 255
    min_value = np.iinfo('uint8').min
    # min_value = 0
    x = x.astype('float64', casting='safe')
    # 将x转化为float类型
    x -= min_value
    x /= ((max_value - min_value) / 2.)
    x -= 1.
    # 2（x - 0）
    # (255 - 0)  - 1
    x = np.sign(x) * (1 / u) * (((1 + u) ** np.abs(x)) - 1)
    # np.sign取数字前面的符号，正数取1,0取0，负数取-1。
    # np.abs返回数字的绝对值
    x = float_to_uint8(x)
    return x


# 浮点数float（0~1）转为（0~255）uint8数据
def float_to_uint8(x):
    x += 1.
    x /= 2.
    uint8_max_value = np.iinfo('uint8').max
    x *= uint8_max_value
    x = x.astype('uint8')
    return x


def predict(desired_sample_rate, predict_initial_input, use_ulaw, fragment_length, predict_seconds):
    run_dir = './'
    # 检查点的目录
    # checkpoint_dir = os.path.join(run_dir, 'checkpoint')
    # 最近一个检查点
    # last_checkpoint = sorted(os.listdir(checkpoint_dir))[-1]
    # 存放生成音频的目录项
    sample_dir = os.path.join(run_dir, 'samples')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    # 利用命名函数给一个生成音频命名
    sample_name = make_sample_name(epoch=2,
                                   predict_use_softmax_as_input=False,
                                   predict_seconds=1,
                                   seed=10,
                                   sample_temperature=None,
                                   sample_argmax=False)
    # 音频存放的名称和路径指定好
    sample_filename = os.path.join(sample_dir, sample_name)
    # 打印把文件存进这个文件目录
    print('Saving to "%s"' % sample_filename)
    # 打开sample_filename.wav文件，配置声道数，量化位数，采样频率
    sample_stream = make_sample_stream(desired_sample_rate, sample_filename)
    # 建立模型，加载之前的检查点，打印模型
    net = model((6000,256))
    # checkpoint_save_path = "./checkpoint/wavenet.ckpt"
    # if os.path.exists(checkpoint_save_path + '.index'):
    #     print('-------------load the model-----------------')
    # net.load_weights(checkpoint_save_path)
    net.summary()
    wav = process_wav(desired_sample_rate, predict_initial_input, use_ulaw)
    # wav 是一个序列采样点序列
    outputs = list(one_hot(wav[:]))
    # write_samples(sample_stream, outputs)这里的sample_stream是一个已经配置好头部的wav后缀的文件
    # warned_repetition = False
    for i in tqdm(xrange(int(desired_sample_rate * predict_seconds))):
        # if not warned_repetition:
        #     if np.argmax(outputs[-1]) == np.argmax(outputs[-2]) and np.argmax(outputs[-2]) == np.argmax(outputs[-3]):
        #         warned_repetition = True
        #         print('Last three predicted outputs where %d' % np.argmax(outputs[-1]))
        #     else:
        #         warned_repetition = False
        prediction_seed = np.expand_dims(np.array(outputs[i:i+fragment_length]), 0)
        # np.expand_dims在相应的轴上扩展维度，axis=0，在第一维上扩展维度
        output = net.predict(prediction_seed)
        # 得到的output是一个（1,fragment_length,256）维的softmax值
        output_dist = output[0][-1]
        # output_dist取得是最后一行output的值
        output_val = draw_sample(output_dist, sample_temperature=None, sample_argmax=True, _rnd=random)
        write_samples(sample_stream, [output_val], True)
    sample_stream.close()
    print("Done!")


# 测试
predict(16000, r'../data/test/Taylor_Swift_-_Welcome_To_New_York.wav', True, 6000, 1)


# if __name__ == '__main__':
  #  main()
