import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
import joblib
from tensorflow.keras.models import load_model
import librosa.display
import tensorflow as tf
from hparams import *
from processing.proc_audio import from_spectro_to_waveform
from config import Tacotron2Config
from tacotron_model import tacotron1

# max_targ_len=Tacotron2Config.MAX_MEL_TIME_LENGTH
config = Tacotron2Config()


# 保存波形到文件
def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))


def evaluate(modelfile, meta_file, eval_dir='./data/', max_targ_len=100):
    metadata = pd.read_csv(meta_file,
                           dtype='object', quoting=3, sep='|',
                           header=None)
    len_train = int(TRAIN_SET_RATIO * len(metadata))
    metadata_testing = metadata.iloc[len_train:]

    # 装入测试数据
    decoder_input_testing = joblib.load(eval_dir + 'decoder_input_testing.pkl')
    # mel_spectro_testing = joblib.load(eval_dir + 'mel_spectro_testing.pkl')
    # spectro_testing = joblib.load(eval_dir + 'spectro_testing.pkl')
    text_input_testing = joblib.load(eval_dir + 'text_input_ml_testing.pkl')
    vocabulary = joblib.load(eval_dir + 'vocabulary.pkl')
    vocab_size = len(vocabulary)
    print("vocab_size", vocab_size)

    # 装入模型
    # saved_model = load_model(modelfile)
    model = tacotron1(vocab_size, config)
    # GO未写
    # _go_frames
    dec_input = tf.expand_dims(tf.zeros(shape=[decoder_input_testing.shape[0], decoder_input_testing.shape[2]]), 1)
    print("dec_input:", dec_input.shape)
    result = 0
    # 预测。这里使用ground truth谱！应该动态解码，利用上一步生成的结果
    # TODO: 自回归解码。解码器每步输入每帧mel谱，产生下一帧mel谱和幅度谱
    # for t in range(1, max_targ_len):
    #     # 将编码器输出 （enc_output） 传送至解码器，解码
    #     mag_output, mel_hat_last_frame,mel_output = model(dec_input, text_input_testing)
    #     if t==1:
    #         result=mag_output
    #     else:
    #         result=tf.concat([result,mag_output],axis=1)
    #
    #     # 使用教师强制，下一步输入符号是训练集中对应目标符号
    #     dec_input = mel_hat_last_frame
    #
    #     print(mel_hat_last_frame.shape)
    mag_result = 0
    for t in range(config.MAX_MEL_TIME_LENGTH):
        mag_output, mel_hat_last_frame, mel_output = model(dec_input, text_input_testing)
        # 将编码器输出 （enc_output） 传送至解码器，解码

        if t == 0:
            mag_result = mag_output
        else:
            mag_result = tf.concat([mag_result, mag_output], axis=1)

        # 使用教师强制，下一步输入符号是训练集中对应目标符号
        dec_input = mel_hat_last_frame
    #

    mag_pred = mag_result[1]
    predicted_audio_item = from_spectro_to_waveform(mag_pred, N_FFT,
                                                    HOP_LENGTH, WIN_LENGTH,
                                                    N_ITER, WINDOW_TYPE,
                                                    MAX_DB, REF_DB, PREEMPHASIS)

    plt.figure(figsize=(14, 5))
    save_wav(predicted_audio_item, eval_dir + 'temp.wav', sr=SAMPLING_RATE)
    # librosa.display.waveplot( , sr=SAMPLING_RATE)
    plt.show()


evaluate('./data/tts-model.h5', 'data/number/metadata.csv', )

#     item_index = 0  # pick any index
#     print('Selected item .wav filename: {}'.format(
#         metadata_testing.iloc[item_index][0]))
#     print('Selected item transcript: {}'.format(
#         metadata_testing.iloc[item_index][1]))
#
#     # predicted_spectro_item = mag_pred[item_index]
#     # predicted_audio_item = from_spectro_to_waveform(predicted_spectro_item, N_FFT,
#     #                                                 HOP_LENGTH, WIN_LENGTH,
#     #                                                 N_ITER, WINDOW_TYPE,
#     #                                                 MAX_DB, REF_DB, PREEMPHASIS)
#
#     plt.figure(figsize=(14, 5))
#     save_wav(predicted_audio_item, eval_dir + 'temp.wav', sr=SAMPLING_RATE)
#     # librosa.display.waveplot( , sr=SAMPLING_RATE)
#     plt.show()
#
#
# evaluate('./data/tts-model.h5', 'data/number/metadata.csv', )
