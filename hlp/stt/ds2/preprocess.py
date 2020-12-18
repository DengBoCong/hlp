
import json

from hlp.stt.ds2.util import get_config
from hlp.stt.utils.load_dataset import load_data
from hlp.stt.utils.audio_process import max_audio_length
from hlp.stt.utils.text_process import split_sentences, get_max_label_length, tokenize_and_encode


if __name__ == "__main__":
    configs = get_config()

    dataset_name = configs["preprocess"]["dataset_name"]
    data_path = configs["train"]["data_path"]
    num_examples = configs["train"]["num_examples"]

    # 获取语料里所有语音路径list和文本list
    print("获得数据集的语音文件和转写...")
    audio_data_path_list, text_list = load_data(dataset_name, data_path, num_examples)

    # 基于文本按照某种mode切分文本
    print("对文本进行切分和编码...")
    mode = configs["preprocess"]["text_process_mode"]
    splitted_text_list = split_sentences(text_list, mode)

    # 将文本处理成对应的token数字序列
    text_int_sequences, tokenizer = tokenize_and_encode(splitted_text_list)

    # 获取音频和文本的最大length，从而进行数据补齐
    audio_feature_type = configs["other"]["audio_feature_type"]
    max_input_length = max_audio_length(audio_data_path_list, audio_feature_type)
    max_label_length = get_max_label_length(text_int_sequences)

    print("保存数据集信息...")
    ds_info_path = configs["preprocess"]["dataset_info_path"]

    dataset_info = {}
    dataset_info["vocab_size"] = len(tokenizer.index_word)
    dataset_info["max_input_length"] = max_input_length
    dataset_info["max_label_length"] = max_label_length
    dataset_info["index_word"] = tokenizer.index_word
    dataset_info["word_index"] = tokenizer.word_index

    with open(ds_info_path, 'w', encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=4)

    print("语音文件数:", len(audio_data_path_list))
    print("vocab_size:", dataset_info["vocab_size"])
    print("最长语音:", dataset_info["max_input_length"])
    print("最长转写文本:", dataset_info["max_label_length"])
