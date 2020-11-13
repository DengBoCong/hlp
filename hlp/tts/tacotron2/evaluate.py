import tensorflow as tf

from config2 import Tacotron2Config
from tacotron2 import load_checkpoint
from prepocesses import dataset_txt, dataset_wave, process_wav_name, map_to_text, get_tokenizer_keras
from tacotron2 import Tacotron2
from audio_process import spec_distance


def evluate(path, csv_dir, save_path_dictionary, vocab_size):
    # 读取测试集的文本数据
    # 统计wav名称
    # a=1代表是number数据集
    # a = 1
    a = 1
    wav_name_list = process_wav_name(path, a)
    # 根据wav名称生成需要的列表
    sentence_list = map_to_text(csv_dir, wav_name_list)
    # 取数据
    input_ids, vocab_inp_size = dataset_txt(sentence_list, save_path_dictionary, "evaluate")
    input_ids = tf.convert_to_tensor(input_ids)

    # 读取测试集的音频数据
    mel_gts, mel_len_wav = dataset_wave(path, config)

    # 模型初始化
    tacotron2 = Tacotron2(vocab_size + 1, config)
    path = config.checkpoingt_dir
    checkpoint = load_checkpoint(tacotron2, path, config)
    print('已恢复至最新的检查点！')
    score_sum = 0

    for i in range(input_ids.shape[0]):
        new_input_ids = input_ids[i]
        new_input_ids = tf.expand_dims(new_input_ids, axis=0)
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = tacotron2.inference(new_input_ids)
        mel2 = mel_gts[i]
        mel2 = tf.expand_dims(mel2, axis=0)
        score = spec_distance(mel_outputs_postnet, mel2)
        score_sum += score
        print('第{}个样本的欧式距离为：{}'.format((i+1), score))
    print("样本平均欧式距离为：", score_sum/input_ids.shape[0])


if __name__ == "__main__":
    config = Tacotron2Config()

    # 字典路径
    save_path_dictionary = config.save_path_dictionary_number
    # 恢复字典
    tokenizer, vocab_size = get_tokenizer_keras(save_path_dictionary)

    # csv文件的路径
    csv_dir = config.csv_dir_number
    # 测试音频文件的路径
    path = config.wave_test_path_number
    evluate(path, csv_dir, save_path_dictionary, vocab_size)
