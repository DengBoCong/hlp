from utils import get_config, get_all_data_path, set_config
import os
from text_process import get_text_list, get_process_text_list, tokenize, get_text_int_sequences, get_max_label_length
import json
from audio_process import get_max_audio_length

#加载数据
def load_data(dataset_name, data_path, train_or_test, num_examples):
    # number语料
    if dataset_name == "number":
        if train_or_test == "train":
            # input_tensor, target_tensor, label_length的训练数据集
            return load_dataset_number(data_path, train_or_test, num_examples)
        else:
            # input_tensor, labels_list的测试数据集
            return load_dataset_number(data_path, train_or_test, num_examples)

# 加载number语料，返回语音文件list和对应文本字符串list
def load_dataset_number(data_path, train_or_test, num_examples = None):
    # 获取配置文件
    configs = get_config()

    # number语料里没有文本集，故先通过audio文件名构建文本文件
    if not os.path.exists(data_path + "/text.txt"):
        files = os.listdir(data_path)
        with open(data_path + "/text.txt", "a") as f:
            for path in files:
                f.write(path[0] + "\n")

    if train_or_test == "train":
        # 获取number语料中训练集语音路径和文本的列表
        text_data_path, audio_path_list = get_all_data_path(data_path)

        audio_data_path_list = [data_path + "/" + audio_path for audio_path in audio_path_list[:num_examples]]
        text_list = get_text_list(data_path + "/" + text_data_path, configs["preprocess"]["text_raw_style"])[:num_examples]
        
        # 基于文本风格和切分方式进行文本处理
        mode = configs["preprocess"]["text_process_mode"]
        process_text_list = get_process_text_list(text_list, mode)

        if configs["train"]["if_is_first_train"]:
            text_int_sequences, tokenizer = tokenize(process_text_list)
            label_length_list = [[len(text_int)] for text_int in text_int_sequences]
            # 获取最长
            max_input_length = get_max_audio_length(audio_data_path_list, configs["other"]["n_mfcc"])
            max_label_length = get_max_label_length(text_int_sequences)

            # 若为初次训练则将构建的字典集合写入json文件
            index_word_path = configs["other"]["index_word_path"]
            with open(index_word_path, 'w', encoding="utf-8") as f:
                json.dump(tokenizer.index_word, f, ensure_ascii=False, indent=4)
            word_index_path = configs["other"]["word_index_path"]
            with open(word_index_path, 'w', encoding="utf-8") as f:
                json.dump(tokenizer.word_index, f, ensure_ascii=False, indent=4)
            
            # 将是否为初次加载设为false
            set_config("train", "if_is_first_train", False)
            set_config("model", "dense_units", len(tokenizer.index_word)+2 )
            set_config("preprocess", "max_input_length", max_input_length)
            set_config("preprocess", "max_label_length", max_label_length)

            return audio_data_path_list, text_int_sequences, label_length_list
        else:
            # 不是初次训练就基于初次训练时写的word_index构建文本
            text_int_sequences = get_text_int_sequences(process_text_list)
            label_length_list = [[len(text_int)] for text_int in text_int_sequences]
            
            return audio_data_path_list, text_int_sequences, label_length_list
    else:
        # 获取number语料中测试集语音路径和文本的列表
        text_data_path, audio_path_list = get_all_data_path(data_path)
        
        audio_data_path_list = [data_path + "/" + audio_path for audio_path in audio_path_list[:num_examples]]
        text_list = get_text_list(data_path + "/" + text_data_path, configs["preprocess"]["text_raw_style"])[:num_examples]
        
        # 测试模块只需要文本字符串list即可
        return audio_data_path_list, text_list



if __name__ == "__main__":
    pass