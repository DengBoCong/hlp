import pandas as pd
import joblib
from processing.proc_text import transform_text_for_ml
from config import Tacotron2Config

config=Tacotron2Config()
def precess_audio(meta_file, output_dir='./data/'):
    metadata = pd.read_csv(meta_file,
                           sep='|', header=None)

    # metadata = metadata.iloc[:10]
    # print(metadata)
    print("metadata[0]:",metadata[0])
    print("metadata[1]:",metadata[1])
    metadata['norm_lower'] = metadata[1].apply(lambda x: x.lower())  # 转写小写

    texts = metadata['norm_lower']
    # print(texts.values)

    # 产生字典
    list_of_existing_chars = list(set(texts.str.cat(sep=' ')))  # 字符词典
    #list_of_existing_chars = list(set(texts))  # 字符词典
    print("list_of_existing_chars：",list_of_existing_chars)
    vocabulary = ''.join(list_of_existing_chars)
    vocabulary += 'P'  # 填充字符
    print("vocabulary：", vocabulary)

    print('vocab size: ', len(vocabulary))
    print(vocabulary)
    # 字符->id
    vocabulary_id = {}
    i = 0
    for char in list(vocabulary):
        vocabulary_id[char] = i
        i += 1

    # 将文本转换成NB_CHARS_MAX长的数字序列，需要补齐
    text_input_ml = transform_text_for_ml(texts.values,
                                          vocabulary_id,
                                          config.NB_CHARS_MAX)

    print(text_input_ml.shape)
    # print(text_input_ml[:3])

    # 划分训练和测试
    len_train = int(config.TRAIN_SET_RATIO * len(metadata))
    text_input_ml_training = text_input_ml[:len_train]
    text_input_ml_testing = text_input_ml[len_train:]

    # save data
    joblib.dump(text_input_ml_training, output_dir + 'text_input_ml_training.pkl')
    joblib.dump(text_input_ml_testing, output_dir + 'text_input_ml_testing.pkl')

    joblib.dump(vocabulary_id, output_dir + 'vocabulary.pkl')

precess_audio('data/number/metadata.csv')