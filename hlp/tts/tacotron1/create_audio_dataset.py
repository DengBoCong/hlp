import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from processing.proc_audio import get_padded_spectros
from config import Tacotron2Config

config=Tacotron2Config()

def precess_audio(meta_file, wave_file_dir, output_dir='./data/'):
    print('Loading the data...')
    metadata = pd.read_csv(meta_file,
                           dtype='object', quoting=3, sep='|', header=None)
    # uncomment this line if you yave weak GPU
    # metadata = metadata.iloc[:500]

    # audio filenames
    dot_wav_filenames = metadata[0].values

    mel_spectro_data = []
    spectro_data = []
    decoder_input = []
    print('Processing the audio samples (computation of spectrograms)...')
    for filename in tqdm(dot_wav_filenames):
        file_path = wave_file_dir + filename
        # 这里的填充只是让时间是r的整数倍
        fname, mel_spectro, spectro = get_padded_spectros(file_path, config.r,
                                                          config.PREEMPHASIS, config.N_FFT,
                                                          config.HOP_LENGTH, config.WIN_LENGTH,
                                                          config.SAMPLING_RATE,
                                                          config.N_MEL, config.REF_DB,
                                                          config.MAX_DB)

        print(mel_spectro.shape, spectro.shape)
        print("mel_spectro[:1, :]：",mel_spectro[:1, :].shape)
        print("mel_spectro[:-1, :]：", mel_spectro[:-1, :].shape)

        decod_inp = np.concatenate((np.zeros_like(mel_spectro[:1, :]),
                                    mel_spectro[:-1, :]), 0)  # mel谱移动一步或一帧
        print('decod_inp: ', decod_inp.shape)
        decod_inp = decod_inp[:, -config.N_MEL:]  # 取后80
        print('decod_inp: ', decod_inp.shape)

        dim0_mel_spectro = mel_spectro.shape[0]
        dim1_mel_spectro = mel_spectro.shape[1]
        padded_mel_spectro = np.zeros((config.MAX_MEL_TIME_LENGTH, dim1_mel_spectro))  # 时间长度一致
        padded_mel_spectro[:dim0_mel_spectro, :dim1_mel_spectro] = mel_spectro
        print('padded_mel_spectro: ', padded_mel_spectro.shape)

        dim0_decod_inp = decod_inp.shape[0]
        dim1_decod_inp = decod_inp.shape[1]
        padded_decod_input = np.zeros((config.MAX_MEL_TIME_LENGTH, dim1_decod_inp))  # 时间长度一致
        padded_decod_input[:dim0_decod_inp, :dim1_decod_inp] = decod_inp
        print('padded_decod_input: ', padded_decod_input.shape)

        dim0_spectro = spectro.shape[0]
        dim1_spectro = spectro.shape[1]
        padded_spectro = np.zeros((config.MAX_MAG_TIME_LENGTH, dim1_spectro))  # 时间长度一致
        padded_spectro[:dim0_spectro, :dim1_spectro] = spectro
        print('padded_spectro: ', padded_spectro.shape)

        mel_spectro_data.append(padded_mel_spectro)
        spectro_data.append(padded_spectro)
        decoder_input.append(padded_decod_input)

    print('Convert into np.array')
    decoder_input_array = np.array(decoder_input)
    print("decoder_input_array:",decoder_input_array.shape)
    mel_spectro_data_array = np.array(mel_spectro_data)
    spectro_data_array = np.array(spectro_data)

    print('Split into training and testing data')
    len_train = int(config.TRAIN_SET_RATIO * len(metadata))

    decoder_input_array_training = decoder_input_array[:len_train]
    decoder_input_array_testing = decoder_input_array[len_train:]
    print("decoder_input_array_training::",decoder_input_array_training.shape)

    mel_spectro_data_array_training = mel_spectro_data_array[:len_train]
    mel_spectro_data_array_testing = mel_spectro_data_array[len_train:]

    spectro_data_array_training = spectro_data_array[:len_train]
    spectro_data_array_testing = spectro_data_array[len_train:]

    print('Save data as pkl')
    joblib.dump(decoder_input_array_training,
                output_dir + 'decoder_input_training.pkl')
    joblib.dump(mel_spectro_data_array_training,
                output_dir + 'mel_spectro_training.pkl')
    joblib.dump(spectro_data_array_training,
                output_dir + 'spectro_training.pkl')

    joblib.dump(decoder_input_array_testing,
                output_dir + 'decoder_input_testing.pkl')
    joblib.dump(mel_spectro_data_array_testing,
                output_dir + 'mel_spectro_testing.pkl')
    joblib.dump(spectro_data_array_testing,
                output_dir + 'spectro_testing.pkl')


precess_audio('data/number/metadata.csv', 'data/number/train01/')