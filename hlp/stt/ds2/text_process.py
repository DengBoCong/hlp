import re
import tensorflow as tf
from utils import get_config


#此方法依据文本是中文文本还是英文文本，若为英文文本是按字符切分还是按单词切分
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

#对英文句子：小写化，切分句子，添加开始和结束标记，将空格转为<space>，按字符切分
def preprocess_en_sentence_char(s):
    s = s.lower().strip()
    result = ""
    for i in s:
        if i == " ":
            result += "<space> "
        else:
            result += i+" "
    result = "<start> " + result.strip() + " <end>"
    return result

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

#获取文本数据list
def get_text_data(data_path, text_data_path, num_examples):
    sentences_list = []
    with open(data_path+"/" + text_data_path, "r") as f:
        sen_list = f.readlines()
    for sentence in sen_list[:num_examples]:
        sentences_list.append(preprocess_sentence(text_row_process(sentence)))
    return sentences_list

#基于数据文本list构建tokenizer
def tokenize(texts):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')  # 无过滤字符
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)  # 文本数字序列
    sequences_length = []
    for seq in sequences:
        sequences_length.append([len(seq)])
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    return sequences,sequences_length,tokenizer