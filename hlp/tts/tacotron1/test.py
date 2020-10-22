import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
import joblib
from tensorflow.keras.models import load_model
import librosa.display

from hparams import *
from processing.proc_audio import from_spectro_to_waveform


# 保存波形到文件
def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))


def evaluate(modelfile, meta_file, eval_dir='./data/'):
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

    # 装入模型
    saved_model = load_model(modelfile)

    # 预测。这里使用ground truth谱！应该动态解码，利用上一步生成的结果
    # TODO: 自回归解码。解码器每步输入每帧mel谱，产生下一帧mel谱和幅度谱
    predictions = saved_model.predict([text_input_testing, decoder_input_testing])

    # mel_pred = predictions[0]  # predicted mel spectrogram
    mag_pred = predictions[1]  # predicted mag spectrogram

    item_index = 0  # pick any index
    print('Selected item .wav filename: {}'.format(
        metadata_testing.iloc[item_index][0]))
    print('Selected item transcript: {}'.format(
        metadata_testing.iloc[item_index][1]))

    predicted_spectro_item = mag_pred[item_index]
    predicted_audio_item = from_spectro_to_waveform(predicted_spectro_item, N_FFT,
                                                    HOP_LENGTH, WIN_LENGTH,
                                                    N_ITER, WINDOW_TYPE,
                                                    MAX_DB, REF_DB, PREEMPHASIS)

    plt.figure(figsize=(14, 5))
    save_wav(predicted_audio_item, eval_dir + 'temp.wav', sr=SAMPLING_RATE)
    librosa.display.waveplot(predicted_audio_item, sr=SAMPLING_RATE)
    plt.show()


evaluate('./data/tts-model.h5', 'data/number/metadata.csv')
