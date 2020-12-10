import tensorflow as tf

from config2 import Tacotron2Config
from tacotron2 import load_checkpoint
from prepocesses import process_wav_name, get_tokenizer_keras
from tacotron2 import Tacotron2
from audio_process import spec_distance
from generator import generator


def evluate(test_data_generator, vocab_size, config, test_batchs):
    # 模型初始化
    tacotron2 = Tacotron2(vocab_size, config)
    path = config.checkpoingt_dir
    checkpoint = load_checkpoint(tacotron2, path, config)
    print('已恢复至最新的检查点！')
    j = 0
    score_sum = 0
    for batch, (input_ids, mel_gts, mel_len_wav) in zip(range(test_batchs), test_data_generator):
        for i in range(input_ids.shape[0]):
            new_input_ids = input_ids[i]
            new_input_ids = tf.expand_dims(new_input_ids, axis=0)
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = tacotron2.inference(new_input_ids)
            mel2 = mel_gts[i]
            mel2 = tf.expand_dims(mel2, axis=0)
            mel2 = tf.transpose(mel2, [0, 2, 1])
            score = spec_distance(mel_outputs_postnet, mel2)
            score_sum += score
            j = j+1
            print('第{}个样本的欧式距离为：{}'.format(j, score))
    print("样本平均欧式距离为：", score_sum/j)


if __name__ == "__main__":
    config = Tacotron2Config()
    batch_size = config.test_batch_size
    # 字典路径
    save_path_dictionary = config.save_path_dictionary
    # 恢复字典
    tokenizer, vocab_size = get_tokenizer_keras(save_path_dictionary)
    # csv文件的路径
    csv_dir = config.csv_dir

    # 测试音频文件的路径
    path = config.wave_test_path
    a = 2
    wav_name_list = process_wav_name(path, a)
    test_batchs = len(wav_name_list)//batch_size

    # 测试生成器
    mode = 'evluate'
    test_data_generator = generator(wav_name_list, batch_size, csv_dir, tokenizer, path, config)
    #评估
    evluate(test_data_generator, vocab_size, config, test_batchs)
