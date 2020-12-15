import os
import numpy as np
import tensorflow as tf
import hlp.tts.utils.data_preprocess as preprocess
from hlp.tts.utils.spec import get_spectrograms


def preprocess_lj_speech_raw_data(metadata_path: str, audio_dir: str, save_path: str, max_length: int,
                                  pre_emphasis: float, n_fft: int, n_mels: int, hop_length: int,
                                  win_length: int, max_db: int, ref_db: int, top_db: int,
                                  spectrum_data_dir: str, audio_suffix: str = ".wav",
                                  tokenized_type: str = "phoneme", cmu_dict_path: str = ""):
    """
    用于处理LJSpeech数据集的方法，将数据整理为<音频地址, 句子>的
    形式，这样方便后续进行分批读取
    :param metadata_path: 元数据CSV文件路径
    :param audio_dir: 音频目录路径
    :param save_path: 保存处理之后的数据路径
    :param max_length: 最大序列长度
    :param audio_suffix: 音频的类型后缀
    :param tokenized_type: 分词类型，默认按音素分词，模式：phoneme(音素)/word(单词)/char(字符)
    :param cmu_dict_path: cmu音素字典路径，使用phoneme时必传
    :param spectrum_data_dir: 保存mel和mag数据目录
    :param pre_emphasis: 预加重
    :param n_fft: FFT窗口大小
    :param n_mels: 产生的梅尔带数
    :param hop_length: 帧移
    :param win_length: 每一帧音频都由window()加窗，窗长win_length，然后用零填充以匹配N_FFT
    :param max_db: 峰值分贝值
    :param ref_db: 参考分贝值
    :param top_db: 峰值以下的阈值分贝值
    :return: 无返回值
    """
    audios_list = os.listdir(audio_dir)
    if not os.path.exists(metadata_path):
        print("元数据CSV文件路径不存在，请检查重试")
        exit(0)

    if not os.path.exists(spectrum_data_dir):
        os.makedirs(spectrum_data_dir)

    count = 0
    with open(metadata_path, 'r', encoding='utf-8') as raw_file, \
            open(save_path, 'w', encoding='utf-8') as save_file:
        for line in raw_file:
            line = line.strip('\n').replace('/', '')
            pair = line.split('|')
            audio_file = pair[0] + audio_suffix
            mel_file = spectrum_data_dir + pair[0] + ".mel.npy"
            mag_file = spectrum_data_dir + pair[0] + ".mag.npy"
            stop_token_file = spectrum_data_dir + pair[0] + ".stop.npy"

            if audios_list.count(audio_file) < 1:
                continue

            text = dispatch_tokenized_func(text=pair[1], tokenized_type=tokenized_type,
                                           cmu_dict_path=cmu_dict_path)
            mel, mag = get_spectrograms(audio_path=audio_dir + audio_file, pre_emphasis=pre_emphasis,
                                        n_fft=n_fft, n_mels=n_mels, hop_length=hop_length,
                                        win_length=win_length, max_db=max_db, ref_db=ref_db, top_db=top_db)
            stop_token = np.zeros(shape=max_length)
            stop_token[len(mel) - 1:] = 1

            mel = tf.keras.preprocessing.sequence.pad_sequences(tf.expand_dims(mel, axis=0),
                                                                maxlen=max_length, dtype="float32", padding="post")
            mel = tf.squeeze(mel, axis=0)
            mel = tf.transpose(mel, [1, 0])

            np.save(file=mel_file, arr=mel)
            np.save(file=mag_file, arr=mag)
            np.save(file=stop_token_file, arr=stop_token)

            save_file.write(mel_file + "\t" + mag_file + "\t" + stop_token_file + "\t" + text + "\n")

            count += 1
            print('\r已处理音频句子对数：{}'.format(count), flush=True, end='')

    print("\n数据处理完毕，共计{}条语音数据".format(count))


def dispatch_tokenized_func(text: str, tokenized_type: str = "phoneme", cmu_dict_path: str = ""):
    """
    用来整合目前所有分词处理方法，通过字典匹配进行调用，默认使用phoneme分词
    :param text: 句子文本
    :param tokenized_type: 分词类型，默认按音素分词，模式：phoneme(音素)/word(单词)/char(字符)
    :param cmu_dict_path: cmu音素字典路径，使用phoneme时必传
    :return: 按照对应方法处理好的文本序列
    """
    operation = {
        "phoneme": lambda: preprocess.text_to_phonemes_converter(text=text,
                                                                 cmu_dict_path=cmu_dict_path),
        "word": lambda: preprocess.text_to_word_converter(text=text),
        "char": lambda: preprocess.text_to_char_converter(text=text)
    }

    return operation.get(tokenized_type, "phoneme")()
