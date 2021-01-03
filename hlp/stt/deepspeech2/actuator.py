import os
import sys
import json
import tensorflow as tf
from argparse import ArgumentParser
sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("\\hlp\\")])
from hlp.stt.utils.pre_treat import dispatch_pre_treat_func
from hlp.stt.utils.utils import load_checkpoint
from hlp.stt.deepspeech2.model import DS2
from hlp.stt.deepspeech2.module import train
from hlp.stt.deepspeech2.module import recognize

if __name__ == '__main__':
    """
    Transformer TTS入口：指令需要附带运行参数
    cmd：python transformer_launch.py -t/--type [执行模式]
    执行类别：pre_treat/train/generate

    generate模式下运行时，输入ESC即退出语音合成
    """
    parser = ArgumentParser(description='transformer tts V1.0.0')
    parser.add_argument('--config_file', default='', type=str, required=False, help='配置文件路径，为空则默认命令行，不为空则使用配置文件参数')
    parser.add_argument('--act', default='pre_treat', type=str, required=False, help='执行类型')
    parser.add_argument('--epochs', default=4, type=int, required=False, help='训练轮次')
    parser.add_argument('--lr', default=0.0001, type=float, required=False, help='学习率')
    parser.add_argument('--conv_layers', default=1, type=int, required=False, help='卷积层数')
    parser.add_argument('--conv_filters', default=256, type=int, required=False, help='卷积层输出空间维度')
    parser.add_argument('--conv_kernel_size', default=11, type=int, required=False, help='卷积核大小')
    parser.add_argument('--conv_strides', default=2, type=int, required=False, help='卷积步幅')
    parser.add_argument('--bi_gru_layers', default=1, type=int, required=False, help='双向GRU层数')
    parser.add_argument('--gru_units', default=256, type=int, required=False, help='GRU单元数')
    parser.add_argument('--fc_units', default=512, type=int, required=False, help='全连接层单元数')
    parser.add_argument('--vocab_size', default=1000, type=int, required=False, help='词汇量大小')
    parser.add_argument('--checkpoint_save_size', default=2, type=int, required=False, help='训练中最多保存checkpoint数量')
    parser.add_argument('--max_sentence_length', default=100, type=int, required=False, help='最大句子序列长度')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='batch大小')
    parser.add_argument('--buffer_size', default=20000, type=int, required=False, help='dataset缓冲区大小')
    parser.add_argument('--beam_size', default=3, type=int, required=False, help='beam_size')
    parser.add_argument('--checkpoint_save_freq', default=2, type=int, required=False, help='检查点保存频率')
    parser.add_argument('--valid_data_split', default=0.1, type=float, required=False, help='从训练数据划分验证数据的比例')
    parser.add_argument('--max_train_data_size', default=0, type=int, required=False, help='从训练集读取最大数据量，0即全部数据')
    parser.add_argument('--max_valid_data_size', default=0, type=int, required=False, help='从验证集集读取最大数据量，0即全部数据')
    parser.add_argument('--audio_feature_type', default='mfcc', type=str, required=False, help='音频处理方式')
    parser.add_argument('--dataset_type', default='thchs30', type=str, required=False, help='数据集类型')
    parser.add_argument('--max_time_step', default=1100, type=int, required=False, help='最大音频补齐长度')
    parser.add_argument('--stop_early_limits', default=5, type=int, required=False, help='验证指标不增长个数停止')
    parser.add_argument('--checkpoint_dir', default='\\data\\checkpoints\\deepspeech2', type=str, required=False,
                        help='检查点保存路径')
    parser.add_argument('--train_file', default='\\data\\data_thchs30\\processed_train_file.txt', type=str,
                        required=False, help='整理后的音频句子对保存路径')
    parser.add_argument('--dict_path', default='\\data\\data_thchs30\\deepspeech2_dict.json', type=str, required=False,
                        help='字典存放路径')
    parser.add_argument('--train_data_dir', default='\\data\\data_thchs30\\data\\', type=str, required=False,
                        help='训练数据集存放目录路径')
    parser.add_argument('--spectrum_data_dir', default='\\data\\data_thchs30\\feature\\', type=str,
                        required=False, help='整理后的音频句子对保存路径')
    parser.add_argument('--save_length_path', default='\\data\\data_thchs30\\length.npy', type=str, required=False,
                        help='训练数据集存放目录路径')
    parser.add_argument('--history_img_path', default='\\data\\history\\deepspeech2\\', type=str,
                        required=False, help='历史数据指标图表保存路径')
    parser.add_argument('--record_path', default='\\data\\record\\', type=str, required=False,
                        help='录音保存目录')

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    # 注意了，有关路径的参数，以deep speech2目录下为基准配置，只要
    # deep speech2目录名未更改，任意移动位置不影响使用
    work_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("\\deepspeech2")]
    execute_type = options['act']

    model = DS2(conv_layers=options['conv_layers'], filters=options['conv_filters'],
                kernel_size=options['conv_kernel_size'], strides=options['conv_strides'],
                bi_gru_layers=options['bi_gru_layers'], gru_units=options['gru_units'],
                fc_units=options['fc_units'], output_dim=options['vocab_size'])
    optimizer = tf.keras.optimizers.Adam()

    ckpt_manager = load_checkpoint(model, checkpoint_dir=work_path + options['checkpoint_dir'],
                                   execute_type=options['act'], checkpoint_save_size=options['checkpoint_save_size'])

    if execute_type == 'train':
        train(epochs=options['epochs'], train_data_path=work_path + options['train_file'],
              max_len=options['max_sentence_length'], vocab_size=options['vocab_size'],
              batch_size=options['batch_size'], buffer_size=options['buffer_size'],
              checkpoint_save_freq=options['checkpoint_save_freq'], checkpoint=ckpt_manager,
              optimizer=optimizer, dict_path=work_path + options['dict_path'],
              valid_data_split=options['valid_data_split'], valid_data_path="",
              max_train_data_size=options['max_train_data_size'], model=model,
              train_length_path=work_path + options['save_length_path'],
              valid_length_path="", stop_early_limits=options['stop_early_limits'],
              max_valid_data_size=options['max_valid_data_size'],
              history_img_path=work_path + options['history_img_path'])
    elif execute_type == 'recognize':
        recognize(model=model, audio_feature_type=options['audio_feature_type'],
                  record_path=work_path + options['record_path'], max_length=options['max_time_step'],
                  dict_path=work_path + options['dict_path'])
    elif execute_type == 'pre_treat':
        dispatch_pre_treat_func(func_type=options['dataset_type'],
                                data_path=work_path + options['train_data_dir'],
                                dataset_infos_file=work_path + options['train_file'],
                                max_length=options['max_time_step'], transcript_row=1,
                                spectrum_data_dir=work_path + options['spectrum_data_dir'],
                                audio_feature_type=options['audio_feature_type'],
                                save_length_path=work_path + options['save_length_path'])
    else:
        parser.error(msg='')
