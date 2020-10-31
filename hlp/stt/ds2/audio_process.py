import wave
import librosa
import pyaudio
from utils import get_config
import tensorflow as tf


#音频的处理
def wav_to_mfcc(wav_path, n_mfcc):
    #加载音频
    y, sr = librosa.load(wav_path, sr=None)
    #提取mfcc(返回list(timestep,n_mfcc))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).transpose(1,0).tolist()
    return mfcc

# 基于语音路径序列，处理成模型的输入tensor
def get_input_tensor(audio_data_path_list, n_mfcc, maxlen):
    mfccs_list = []
    for audio_path in audio_data_path_list:
        mfcc = wav_to_mfcc(audio_path, n_mfcc)
        mfccs_list.append(mfcc)

    mfccs_numpy = tf.keras.preprocessing.sequence.pad_sequences(
        mfccs_list,
        maxlen=maxlen,
        padding='post',
        dtype='float32'
        )
    input_tensor = tf.convert_to_tensor(mfccs_numpy)

    return input_tensor

#获取最长的音频length(timesteps)
def get_max_audio_length(audio_data_path_list, n_mfcc):

    max_audio_length = 0
    for audio_path in audio_data_path_list:
        mfcc = wav_to_mfcc(audio_path, n_mfcc)
        max_audio_length = max(max_audio_length, len(mfcc))
    
    return max_audio_length

# 获取麦克风录音并保存在filepath中
def record():
    configs = get_config()
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # 声道数
    RATE = 16000  # 采样率
    RECORD_SECONDS = int(input("录音时长(秒):"))
    WAVE_OUTPUT_FILENAME = configs["record"]["record_path"]
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