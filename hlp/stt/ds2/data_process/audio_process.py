import wave
import pyaudio
import tensorflow as tf

import sys
sys.path.append("..")
from utils.features import wav_to_feature

# 基于语音路径序列，处理成模型的输入tensor,以及获取输入的时间步长
def get_input_and_length(audio_data_path_list, audio_feature_type, maxlen):
    audio_feature_list = []
    input_length_list = []
    for audio_path in audio_data_path_list:
        audio_feature = wav_to_feature(audio_path, audio_feature_type)
        audio_feature_list.append(audio_feature)
        input_length_list.append([len(audio_feature)])

    audio_feature_numpy = tf.keras.preprocessing.sequence.pad_sequences(
        audio_feature_list,
        maxlen = maxlen,
        dtype = 'float32',
        padding = 'post'
        )
    input_tensor = tf.convert_to_tensor(audio_feature_numpy)
    input_length = tf.convert_to_tensor(input_length_list)

    return input_tensor, input_length

#获取最长的音频length(timesteps)
def get_max_audio_length(audio_data_path_list, audio_feature_type):
    max_audio_length = 0

    for audio_path in audio_data_path_list:
        audio_feature = wav_to_feature(audio_path, audio_feature_type)
        max_audio_length = max(max_audio_length, len(audio_feature))
    
    return max_audio_length

# 获取麦克风录音并保存在filepath中
def record(record_path, record_duration):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # 声道数
    RATE = 16000  # 采样率
    RECORD_SECONDS = record_duration
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