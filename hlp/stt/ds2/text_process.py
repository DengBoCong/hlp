import re
import tensorflow as tf
from utils import get_config
import tensorflow as tf


#此方法依据文本是中文文本还是英文文本，若为英文文本是按字符切分还是按单词切分
def preprocess_sentence(str):
    configs = get_config()
    mode = configs["preprocess"]["text_process_mode"]
    if mode == "cn":
        return preprocess_sentence_ch(str)
    elif mode == "en_word":
        return preprocess_sentence_en_word(str)
    else:
        return preprocess_sentence_en_char(str)

#基于数据文本规则的行获取
def text_row_process(str):
    configs = get_config()
    style = configs["preprocess"]["text_raw_style"]
    if style == 1:
        #当前数据文本的每行为'index string\n'
        return str.strip().split(" ",1)[1].lower()
    elif style == 2:
        #当前数据文本的每行为'index\tstring\n'
        return str.strip().split("\t",1)[1].lower()
    else:
        #文本每行"string\n"
        return str.strip().lower()

#word_list转token序列
def text_to_int_sequence(word_list, word_index):
    int_sequence = []
    for c in word_list:
        int_sequence.append(int(word_index[c]))
    return int_sequence

#基于每个生成器构建模型所需的文本数据
def get_text_label(data_path, text_data_path, i, batch_size, word_index):
    configs = get_config()
    max_targets_len = configs["preprocess"]["max_targets_len"]
    
    target_list = []
    target_length_list = []
    with open(data_path + "/" + text_data_path, "r") as f:
        sentence_list = f.readlines()
    for sentence in sentence_list[i*batch_size : (i+1)*batch_size]:
        sen_word_list = preprocess_sentence(text_row_process(sentence)).split(" ")
        sen_int_list = text_to_int_sequence(sen_word_list, word_index)
        target_length_list.append([len(sen_word_list)])
        target_list.append(sen_int_list)
    
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        target_list,
        maxlen=max_targets_len,
        padding='post'
        )
    target_length = tf.convert_to_tensor(target_length_list)
    return target_tensor, target_length

#获取测试时所需的文本list
def get_lables_list(data_path, text_data_path, i, batch_size):
    labels_list = []
    with open(data_path + "/" + text_data_path, "r") as f:
        sentence_list = f.readlines()
    for sentence in sentence_list[i*batch_size : (i+1)*batch_size]:
        labels_list.append(text_row_process(sentence))
    return labels_list

#预处理文本数据后得到index_word,word_index,max_label_length
def get_text_tokenizer(folder_path, text_data_path, num_examples):
    max_label_length = 0
    word_set = set()
    with open(folder_path + "/" + text_data_path, "r") as f:
        sentence_list = f.readlines()
    for sentence in sentence_list[:num_examples]:
        sen_word_list = preprocess_sentence(text_row_process(sentence)).split(" ")
        max_label_length = max(max_label_length, len(sen_word_list))
        word_set = word_set | set(sen_word_list)
    
    word_list = list(word_set)
    index_word = dict(zip(range(1, len(word_list)+1), word_list))
    word_index = dict(zip(word_list, range(1, len(word_list)+1)))
    
    return index_word, word_index, max_label_length

# 对英文句子：小写化，切分句子，添加开始和结束标记，按单词切分
def preprocess_sentence_en_word(s):
    s = s.lower().strip()
    # 在单词与跟在其后的标点符号之间插入一个空格
    # 例如： "he is a boy." => "he is a boy ."
    # 参考：https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    s = re.sub(r"([?.!,])", r" \1 ", s)  # 切分断句的标点符号
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格

    # 除了 (a-z, A-Z, ".", "?", "!", ",")，将所有字符替换为空格
    s = re.sub(r"[^a-zA-Z?.!,]+", " ", s)
    s = s.strip()
    """
    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    s = '<start> ' + s + ' <end>'
    """
    return s

#对英文句子：小写化，切分句子，添加开始和结束标记，将空格转为<space>，按字符切分
def preprocess_sentence_en_char(s):
    s = s.lower().strip()

    result = ""
    for i in s:
        if i == " ":
            result += "<space> "
        else:
            result += i + " "
    """
    result = "<start> " + result.strip() + " <end>"
    """
    return result.strip()

# 对中文句子：按字切分句子，添加开始和结束标记
def preprocess_sentence_ch(s):
    s = s.lower().strip()

    s = [c for c in s]
    s = ' '.join(s)
    s = re.sub(r'[" "]+', " ", s)  # 合并多个空格

    s = s.strip()
    """
    # 给句子加上开始和结束标记
    # 以便模型知道何时开始和结束预测
    s = '<start> ' + s + ' <end>'
    """
    return s