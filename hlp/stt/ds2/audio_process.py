import librosa
import pyaudio
from utils import get_config
import wave


#音频的处理
def wav_to_mfcc(wav_path):
    configs = get_config()
    n_mfcc = configs["other"]["n_mfcc"]
    #加载音频
    y, sr = librosa.load(wav_path,sr=None)
    #提取mfcc(返回list(timestep,n_mfcc))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).transpose(1,0).tolist()
    return mfcc

#获取音频特征mfccs list
def get_audio_feature(data_path,audio_data_path_list,num_examples):
    mfccs_list = []
    for audio_path in audio_data_path_list[:num_examples]:
        mfcc = wav_to_mfcc(data_path + "/" + audio_path)
        mfccs_list.append(mfcc)
    return mfccs_list

# 获取麦克风录音并保存在filepath中
def record(file_path):
    CHUNK = 256
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # 声道数
    RATE = 16000  # 采样率
    configs = get_config()
    RECORD_SECONDS = configs["record"]["record_times"]  # 录音时长
    WAVE_OUTPUT_FILENAME = file_path
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("开始录音：请在%d秒内输入语音:" % (RECORD_SECONDS))
    frames = []
    for i in range(1, int(RATE / CHUNK * RECORD_SECONDS) + 1):
        data = stream.read(CHUNK)
        frames.append(data)
        if (i % (RATE / CHUNK)) == 0:
            print('\r%s%d%s' % ("剩余", int(RECORD_SECONDS - (i // (RATE / CHUNK))), "秒"), end="")
    print("\n录音结束\n")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()