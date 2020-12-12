import os
import time
import tensorflow as tf

from config2 import Tacotron2Config
from tacotron2 import load_checkpoint
from prepocesses import dataset_txt, tar_stop_token, process_wav_name, map_to_text, get_tokenizer_keras
from tacotron2 import Tacotron2
from generator import generator


if __name__ == "__main__":
    config = Tacotron2Config()
    batch = config.batch_size
    # 检查点
    checkpoint_dir = config.checkpoingt_dir

    # 如果要用number数据集那么输入如下：
    # # 设置文件路径，训练文本与音频路径
    # wave_train_path = config.wave_train_path_number
    # # 保存字典路径
    # save_path_dictionary = config.save_path_dictionary_number
    # # csv文件的路径
    # csv_dir = config.csv_dir_number
    # # a=1代表他是number数据集
    # a = 1

    # 如果要用ljspeech数据集那么输入如下：
    # 设置文件路径，训练文本与音频路径
    wave_train_path = config.wave_train_path
    # 保存字典路径
    save_path_dictionary = config.save_path_dictionary
    # csv文件的路径
    csv_dir = config.csv_dir
    #a=1代表他是number数据集等于其他表示他是ljspeech
    a = 2
    batch_size = config.batch_size

    # 统计wav名称
    wav_name_list = process_wav_name(wave_train_path, a)
    batchs = len(wav_name_list)//batch_size

    #获取所有文本，以用来制作字典
    sentence_list = map_to_text(csv_dir, wav_name_list)

    #保存字典，获取vocab_size
    en_seqs, vocab_size = dataset_txt(sentence_list, save_path_dictionary, config)
    tokenizer, vocab_size = get_tokenizer_keras(save_path_dictionary)
    mode = 'train'

    #生成器初始化
    train_data_generator = generator(wav_name_list, batch_size, csv_dir, tokenizer, wave_train_path, config)

    # 初始化模型和优化器
    tacotron2 = Tacotron2(vocab_size, config)
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)

    # 如果检查点存在就恢复，如果不存在就重新创建一个
    if os.path.exists(checkpoint_dir):
        ckpt_manager = load_checkpoint(tacotron2, checkpoint_dir, config)
        #checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        print('已恢复至最新的检查点！')
    else:
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(tacotron2=tacotron2)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=config.max_to_keep)
        print('新的检查点已准备创建！')

    epochs = 100
    mel_outputs = train(tacotron2, optimizer, train_data_generator, epochs, ckpt_manager, batchs)
