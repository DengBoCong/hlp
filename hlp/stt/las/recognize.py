# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 19:55:49 2020

@author: 九童
交互式语音识别
"""
# -*- coding: utf-8 -*-
import os
import wave
import pyaudio
from tqdm import tqdm
import tensorflow as tf
from hlp.stt.las import train
from hlp.stt.las.model import las
from hlp.stt.las.data_processing import librosa_mfcc


def record_audio(wave_out_path, record_second):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    wf = wave.open(wave_out_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    print("* recording")
    for i in tqdm(range(0, int(RATE / CHUNK * record_second))):
        data = stream.read(CHUNK)
        wf.writeframes(data)
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()


def recognition(wav_path):
    test_path = ".\\data\\wav_test"

    # 测试集文本标签
    test_path_to_file = ".\\data\\data_test.txt"
    # 每一步mfcc所取得特征数
    n_mfcc = 39
    test_num = 80
    embedding_dim = 256
    units = 512
    BATCH_SIZE = 1  # 只支持BATCH_SIZE为1的评估
    steps_per_epoch, test_targ_tokenizer, test_max_length_targ, test_max_length_inp, _ = train.create_dataset(test_path,
                                                                                                              test_path_to_file,
                                                                                                              test_num,
                                                                                                              n_mfcc,
                                                                                                              BATCH_SIZE)
    test_vocab_tar_size = len(test_targ_tokenizer.word_index) + 1  # 含填充的0
    optimizer = tf.keras.optimizers.Adam()

    # vocab_tar_size应从配置文件中得到，此处暂用test_vocab_tar_size代替试试
    model = las.las_model(test_vocab_tar_size, embedding_dim, units, BATCH_SIZE)

    # 检查点
    checkpoint_dir = './checkpoints'
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    # 恢复检查点目录 （checkpoint_dir） 中最新的检查点
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


    hidden = tf.zeros((BATCH_SIZE, units))
    wav_tensor = librosa_mfcc.wav_to_mfcc(wav_path, n_mfcc)
    dec_input = tf.expand_dims([test_targ_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
    result = ''  # 识别结果字符串

    for t in range(test_max_length_targ):  # 逐步解码或预测
        predictions, dec_hidden = model(wav_tensor, hidden, dec_input)
        predicted_ids = tf.argmax(predictions, 1).numpy()  # 贪婪解码，取最大 
        result += test_targ_tokenizer.index_word[predicted_ids[0]]  # 目标句子
        if test_targ_tokenizer.index_word[predicted_ids[0]] == '<end>':
            break
        # 预测的 ID 被输送回模型            
        dec_input = tf.expand_dims(predicted_ids, 1)
    print('****************************')
    print('Speech recognition results=====================: {}'.format(result))


if __name__ == "__main__":
    record_audio("output.wav", record_second=2)
    file_path = ".\\"
    recognition(file_path)
