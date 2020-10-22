import io
import os
import re
import tensorflow as tf
import librosa
import numpy as np
import json
import pyaudio
import wave


#获取配置文件
def get_config():
    with open("config.json","r",encoding="utf-8") as f:
        configs = json.load(f)
    return configs

def set_config(config_class,key,value):
    configs = get_config()
    configs[config_class][key]=value
    with open("config.json",'w',encoding="utf-8") as f:
        json.dump(configs, f, ensure_ascii=False, indent=4)

def get_config_model():
    configs = get_config()
    return configs["other"]["n_mfcc"],configs["model"]["conv_layers"],configs["model"]["conv_filters"],configs["model"]["conv_kernel_size"],configs["model"]["conv_strides"],configs["model"]["bi_gru_layers"],configs["model"]["gru_units"],configs["model"]["dense_units"]

#根据数据文件夹名获取所有的文件名，包括文本文件名和音频文件名列表
def get_all_data_path(data_path):
    #data_path是数据文件夹的路径
    files = os.listdir(data_path) #得到数据文件夹下的所有文件名称list
    text_data_path = files.pop()
    audio_data_path_list = files
    return text_data_path,audio_data_path_list

# 对英文句子：小写化，切分句子，添加开始和结束标记，按单词切分
def preprocess_en_sentence_word(s):
    s = s.lower().strip()

    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    # 参考：https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    s = re.sub(r"([?.!,])", r" \1 ", s)  # 切分断句的标点符号
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格

    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    s = re.sub(r"[^a-zA-Z?.!,]+", " ", s)

    s = s.strip()

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    s = '<start> ' + s + ' <end>'
    return s

#对英文句子：小写化，切分句子，添加开始和结束标记，按字符切分
def preprocess_en_sentence_char(s):
    #...
    return s

# 对中文句子：按字切分句子，添加开始和结束标记
def preprocess_ch_sentence(s):
    s = s.lower().strip()

    s = [c for c in s]
    s = ' '.join(s)
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格

    s = s.strip()

    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    s = '<start> ' + s + ' <end>'
    return s

#此方法依据文本是中文文本还是英文文本，且英文文本是按字符切分还是按单词切分
def preprocess_sentence(str):
    configs = get_config()
    mode = configs["preprocess"]["text_process_mode"]
    if mode == "cn":
        return preprocess_ch_sentence(str)
    elif mode == "en_word":
        return preprocess_en_sentence_word(str)
    else:
        return preprocess_en_sentence_char(str)

#基于数据文本规则的行获取
def text_row_process(str):
    configs = get_config()
    style = configs["preprocess"]["text_raw_style"]
    if style == 1:
        #当前数据文本的每行为'index string\n',且依据英文单词切分
        return str.strip().split(" ",1)[1].lower()
    else:
        #文本每行"string\n"
        return str.strip().lower()

#获取文本数据list
def get_text_data(data_path,text_data_path,num_examples):
    sentences_list = []
    with open(data_path+"/"+text_data_path,"r") as f:
        sen_list = f.readlines()
    for sentence in sen_list[:num_examples]:
        sentences_list.append(preprocess_sentence(text_row_process(sentence)))
    return sentences_list

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
        mfcc = wav_to_mfcc(data_path+"/"+audio_path)
        mfccs_list.append(mfcc)
    return mfccs_list

def create_dataset(data_path,text_data_path,audio_data_path_list,num_examples):
    mfccs_list = get_audio_feature(data_path,audio_data_path_list,num_examples)
    sentences_list = get_text_data(data_path,text_data_path,num_examples)
    return mfccs_list,sentences_list

def tokenize(texts):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')  # 无过滤字符
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列
    sequences_length = []
    for seq in sequences:
        sequences_length.append([len(seq)])
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,padding='post')
    return sequences,sequences_length,tokenizer

def get_index_word():
    configs = get_config()
    index_word_json_path = configs["other"]["index_word_json_path"]
    with open(index_word_json_path,"r",encoding="utf-8") as f:
        index_word = json.load(f)
    return index_word

def convert(tokenizer, tensor):
    for t in tensor:
        if t!=0:
            print ("%d ----> %s" % (t, tokenizer.index_word[t]))

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

# 输入的两个参数均是字符串的list,是wer计算的入口
def wers(originals, results):
    count = len(originals)
    try:
        assert count > 0
    except:
        print(originals)
        raise ("ERROR assert count>0 - looks like data is missing")
    rates = []
    mean = 0.0
    assert count == len(results)
    for i in range(count):
        rate = _wer(originals[i], results[i])
        mean = mean + rate
        rates.append(rate)
    return rates, mean / float(count)

def _wer(original, result):
    r"""
    The WER is defined as the editing/Levenshtein distance on word level
    divided by the amount of words in the original text.
    In case of the original having more words (N) than the result and both
    being totally different (all N words resulting in 1 edit operation each),
    the WER will always be 1 (N / N = 1).
    """
    # The WER ist calculated on word (and NOT on character) level.
    # Therefore we split the strings into words first:
    original = original.split()
    result = result.split()
    return _levenshtein(original, result) / float(len(original))

def lers(originals, results):
    count = len(originals)
    assert count > 0
    rates = []
    norm_rates = []

    mean = 0.0
    norm_mean = 0.0

    assert count == len(results)
    for i in range(count):
        rate = _levenshtein(originals[i], results[i])
        mean = mean + rate

        normrate = (float(rate) / len(originals[i]))

        norm_mean = norm_mean + normrate

        rates.append(rate)
        norm_rates.append(round(normrate, 4))

    return rates, (mean / float(count)), norm_rates, (norm_mean / float(count))

def _levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def decode_output(seq,index_word):
    configs = get_config()
    mode = configs["preprocess"]["text_process_mode"]
    if mode == "cn":
        return decode_output_ch_sentence(seq,index_word)
    elif mode == "en_word":
        return decode_output_en_sentence_word(seq,index_word)
    else:
        return decode_output_en_sentence_char(seq,index_word)

def decode_output_ch_sentence(seq,index_word):
    result = ""
    for i in seq:
        if i>=1 and i<=len(index_word):
            word = index_word[str(i)]
            if word != "<start>":
                if word != "<end>":
                    result += word
                else:
                    return result
    return result

def decode_output_en_sentence_word(seq,index_word):
    result = ""
    for i in seq:
        if i>=1 and i<=(len(index_word)):
            word = index_word[str(i)]
            if word != "<start>":
                if word != "<end>":
                    result += word+" "
                else:
                    return result
    return result

def decode_output_en_sentence_char(seq,index_word):
    pass


if __name__ == "__main__":
    pass