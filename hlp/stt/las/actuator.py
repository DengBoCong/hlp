import os
import sys
import json
import tensorflow as tf
from argparse import ArgumentParser
sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("\\hlp\\")])
from hlp.stt.utils.pre_treat import dispatch_pre_treat_func
from hlp.stt.las.module import train
from hlp.stt.las.module import load_checkpoint
from hlp.stt.las.model.las import LAS
from hlp.stt.las.model.plas import PLAS


def main():
    parser = ArgumentParser(description='transformer tts V1.0.0')
    parser.add_argument('--config_file', default='', type=str, required=False, help='配置文件路径，为空则默认命令行，不为空则使用配置文件参数')
    parser.add_argument('--act', default='pre_treat', type=str, required=False, help='执行类型')
    parser.add_argument('--dataset_type', default='thchs30', type=str, required=False, help='数据集类型')
    parser.add_argument('--model_type', default='las_d_w', type=str, required=False, help='模型类型')
    parser.add_argument('--epochs', default=4, type=int, required=False, help='训练轮次')
    parser.add_argument('--vocab_size', default=1000, type=int, required=False, help='词汇量大小')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='batch大小')
    parser.add_argument('--buffer_size', default=20000, type=int, required=False, help='dataset缓冲区大小')
    parser.add_argument('--lr', default=0.0001, type=float, required=False, help='学习率')
    parser.add_argument('--embedding_dim', default=128, type=int, required=False, help='嵌入层维度')
    parser.add_argument('--units', default=256, type=int, required=False, help='单元数')
    parser.add_argument('--cnn1_filters', default=16, type=int, required=False, help='卷积输出空间维度')
    parser.add_argument('--cnn1_kernel_size', default=3, type=int, required=False, help='卷积核大小')
    parser.add_argument('--cnn2_filters', default=16, type=int, required=False, help='卷积输出空间维度')
    parser.add_argument('--cnn2_kernel_size', default=3, type=int, required=False, help='卷积核大小')
    parser.add_argument('--max_pool_strides', default=2, type=int, required=False, help='池化步幅')
    parser.add_argument('--max_pool_size', default=7, type=int, required=False, help='池化大小')
    parser.add_argument('--d', default=1, type=int, required=False, help='')
    parser.add_argument('--w', default=256, type=int, required=False, help='')
    parser.add_argument('--emb_dim', default=256, type=int, required=False, help='')
    parser.add_argument('--dec_units', default=256, type=int, required=False, help='')
    parser.add_argument('--max_time_step', default=1100, type=int, required=False, help='最大音频补齐长度')
    parser.add_argument('--max_sentence_length', default=100, type=int, required=False, help='最大句子序列长度')
    parser.add_argument('--checkpoint_save_freq', default=2, type=int, required=False, help='检查点保存频率')
    parser.add_argument('--valid_data_split', default=0.2, type=float, required=False, help='从训练数据划分验证数据的比例')
    parser.add_argument('--max_train_data_size', default=20, type=int, required=False, help='从训练集读取最大数据量，0即全部数据')
    parser.add_argument('--max_valid_data_size', default=0, type=int, required=False, help='从验证集集读取最大数据量，0即全部数据')
    parser.add_argument('--checkpoint_save_size', default=2, type=int, required=False, help='训练中最多保存checkpoint数量')
    parser.add_argument('--audio_feature_type', default='mfcc', type=str, required=False, help='音频处理方式')
    parser.add_argument('--train_data_dir', default='\\data\\data_thchs30\\data\\', type=str, required=False,
                        help='训练数据集存放目录路径')
    parser.add_argument('--train_file', default='\\data\\data_thchs30\\processed_train_file.txt', type=str,
                        required=False, help='整理后的音频句子对保存路径')
    parser.add_argument('--spectrum_data_dir', default='\\data\\data_thchs30\\feature\\', type=str,
                        required=False, help='整理后的音频句子对保存路径')
    parser.add_argument('--dict_path', default='\\data\\data_thchs30\\las_dict.json', type=str, required=False,
                        help='字典存放路径')
    parser.add_argument('--checkpoint_dir', default='\\data\\checkpoints\\las', type=str, required=False,
                        help='检查点保存路径')

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    # 注意了，有关路径的参数，以las目录下为基准配置，只要
    # las目录名未更改，任意移动位置不影响使用
    work_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("\\las")]
    execute_type = options['act']
    model_type = options['model_type']

    # 选择模型类型
    if model_type == "las":
        model = PLAS(vocab_tar_size=options['vocab_size'], embedding_dim=options['embedding_dim'],
                     units=options['units'], batch_size=options['batch_size'])
    elif model_type == "las_d_w":
        model = LAS(vocab_tar_size=options['vocab_size'], cnn1_filters=options['cnn1_filters'],
                    cnn1_kernel_size=options['cnn1_kernel_size'], cnn2_filters=options['cnn2_filters'],
                    cnn2_kernel_size=options['cnn2_kernel_size'], max_pool_strides=options['max_pool_strides'],
                    max_pool_size=options['max_pool_size'], d=options['d'], w=options['w'],
                    embedding_dim=options['emb_dim'], dec_units=options['dec_units'],
                    batch_size=options['batch_size'])

    optimizer = tf.keras.optimizers.Adam(lr=options['lr'])
    ckpt_manager = load_checkpoint(model=model, checkpoint_dir=work_path + options['checkpoint_dir'],
                                   execute_type=options['act'], checkpoint_save_size=options['checkpoint_save_size'])

    if execute_type == 'train':
        train(epochs=options['epochs'], train_data_path=work_path + options['train_file'],
              max_len=options['max_sentence_length'], vocab_size=options['vocab_size'],
              batch_size=options['batch_size'], buffer_size=options['buffer_size'],
              checkpoint_save_freq=options['checkpoint_save_freq'], checkpoint=ckpt_manager,
              optimizer=optimizer, dict_path=work_path + options['dict_path'], model=model,
              valid_data_split=options['valid_data_split'], valid_data_path="",
              max_train_data_size=options['max_train_data_size'],
              max_valid_data_size=options['max_valid_data_size'])
    elif execute_type == 'recognize':
        print("待完善")
    elif execute_type == 'pre_treat':
        dispatch_pre_treat_func(func_type=options['dataset_type'],
                                data_path=work_path + options['train_data_dir'],
                                dataset_infos_file=work_path + options['train_file'],
                                max_length=options['max_time_step'], transcript_row=0,
                                spectrum_data_dir=work_path + options['spectrum_data_dir'],
                                audio_feature_type=options['audio_feature_type'])
    else:
        parser.error(msg='')


if __name__ == '__main__':
    """
    LAS STT入口：指令需要附带运行参数
    cmd：python transformer_launch.py -t/--type [执行模式]
    执行类别：pre_treat/train/generate

    generate模式下运行时，输入ESC即退出语音合成
    """
    main()
