import json
from data_process.load_dataset import load_data
from data_process.audio_process import get_max_audio_length
from data_process.text_process import get_process_text_list, get_max_label_length, tokenize

from util import get_config


if __name__ == "__main__":
    configs = get_config()

    dataset_name = configs["preprocess"]["dataset_name"]
    data_path = configs["train"]["data_path"]
    text_row_style = configs["preprocess"]["text_row_style"]
    num_examples = configs["train"]["num_examples"]

    # 获取语料里所有语音路径list和文本list
    audio_data_path_list, text_list = load_data(dataset_name, data_path, text_row_style, num_examples)

    # 基于文本按照某种mode切分文本
    mode = configs["preprocess"]["text_process_mode"]
    process_text_list = get_process_text_list(text_list, mode)

    # 将文本处理成对应的token数字序列
    text_int_sequences, tokenizer = tokenize(process_text_list)

    # 获取音频和文本的最大length，从而进行数据补齐
    audio_feature_type = configs["other"]["audio_feature_type"]
    max_input_length = get_max_audio_length(audio_data_path_list, audio_feature_type)
    max_label_length = get_max_label_length(text_int_sequences)

    # 将数据集的相关信息写入dataset_information.json文件
    dataset_information_path = configs["preprocess"]["dataset_information_path"]
    
    dataset_information = {}
    dataset_information["vocab_size"] = len(tokenizer.index_word)
    dataset_information["max_input_length"] = max_input_length
    dataset_information["max_label_length"] = max_label_length
    dataset_information["index_word"] = tokenizer.index_word
    dataset_information["word_index"] = tokenizer.word_index

    with open(dataset_information_path, 'w', encoding="utf-8") as f:
        json.dump(dataset_information, f, ensure_ascii=False, indent=4)

    print("语音文件数:", num_examples)
    print("vocab_size:", dataset_information["vocab_size"])
    print("最长语音:", dataset_information["max_input_length"])
    print("最长转写文本:", dataset_information["max_label_length"])