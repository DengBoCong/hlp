import wave
import pyaudio
import tensorflow as tf
from python_speech_features import mfcc, logfbank, delta
from scipy.io import wavfile
import numpy as np


def wav_to_feature(wav_path, audio_feature_type):
    fs, audio = wavfile.read(wav_path)
    
    if audio_feature_type == "mfcc":
        return get_mfcc_feature(audio, fs)
    elif audio_feature_type == "fbank":
        return get_fbank_feature(audio, fs)

def get_mfcc_feature(wavsignal, fs):
    # 输入为wav文件数学表示和采样频率，输出为语音的MFCC特征(默认13维)+一阶差分+二阶差分；
    feat_mfcc = mfcc(wavsignal, fs)
    feat_mfcc_d = delta(feat_mfcc, 2)
    feat_mfcc_dd = delta(feat_mfcc_d, 2)
    
    # (timestep, 39)
    wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
    return wav_feature.tolist()

def get_fbank_feature(wavsignal, fs):
    # 输入为wav文件数学表示和采样频率，输出为语音的FBANK特征
    feat_fbank = logfbank(wavsignal, fs, nfilt=80)
    
    # 未加差分, (timestep, 80)
    wav_feature = np.column_stack((feat_fbank))
    return wav_feature.tolist()

# 基于语音路径序列，处理成模型的输入tensor
def get_input_tensor(audio_data_path_list, audio_feature_type, maxlen):
    audio_feature_list = []
    for audio_path in audio_data_path_list:
        audio_feature = wav_to_feature(audio_path, audio_feature_type)
        audio_feature_list.append(audio_feature)

    audio_feature_numpy = tf.keras.preprocessing.sequence.pad_sequences(
        audio_feature_list,
        maxlen=maxlen,
        padding='post',
        dtype='float32'
        )
    input_tensor = tf.convert_to_tensor(audio_feature_numpy)

    return input_tensor

#获取最长的音频length(timesteps)
def get_max_audio_length(audio_data_path_list, audio_feature_type):
    max_audio_length = 0

    for audio_path in audio_data_path_list:
        audio_feature = wav_to_feature(audio_path, audio_feature_type)
        max_audio_length = max(max_audio_length, len(audio_feature))
    
    return max_audio_length

# 获取麦克风录音并保存在filepath中
def record(record_path):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # 声道数
    RATE = 16000  # 采样率
    RECORD_SECONDS = int(input("录音时长(秒):"))
    WAVE_OUTPUT_FILENAME = record_path
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    print("开始录音：请在%d秒内输入语音:" % (RECORD_SECONDS))
    for i in range(1, int(RATE / CHUNK * RECORD_SECONDS) + 1):
        data = stream.read(CHUNK)
        frames.append(data)
        if (i % (RATE / CHUNK)) == 0:
            print('\r%s%d%s' % ("剩余", int(RECORD_SECONDS - (i // (RATE / CHUNK))), "秒"), end="")
    print("\r录音结束\n")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


if __name__ == "__main__":
    pass