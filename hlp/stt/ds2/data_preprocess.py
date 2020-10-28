from utils import get_config, set_config
from audio_process import get_max_audio_length
from text_process import get_text_tokenizer
import json
from utils import get_all_data_path


# 将数据生成的字典集写入json,将对应的模型dense层单元数和最大的输入长度写入config.json,返回word_index
def data_preprocess(folder_path, num_examples):
    #获取文本和音频的数据路径
    text_data_path, audio_data_path_list = get_all_data_path(folder_path)
    
    max_audio_length = get_max_audio_length(folder_path, audio_data_path_list, num_examples)
    index_word, word_index, max_label_length = get_text_tokenizer(folder_path, text_data_path, num_examples)

    #保存index_word和word_index到json文件
    configs = get_config()
    index_word_path = configs["other"]["index_word_path"]
    with open(index_word_path, 'w', encoding="utf-8") as f:
        json.dump(index_word, f, ensure_ascii=False, indent=4)
    word_index_path = configs["other"]["word_index_path"]
    with open(word_index_path, 'w', encoding="utf-8") as f:
        json.dump(word_index, f, ensure_ascii=False, indent=4)
    
    #基于训练数据将输入序列的长度以及模型dense层所需的单元数(即token集合数量)写入配置文件
    set_config("preprocess", "max_inputs_len", max_audio_length)
    set_config("preprocess", "max_targets_len", max_label_length)
    set_config("model", "dense_units", len(index_word) + 2)


if __name__ == '__main__':
    configs = get_config()
    data_path = configs["train"]["data_path"]
    num_examples = configs["train"]["num_examples"]
    data_preprocess(data_path, num_examples)
