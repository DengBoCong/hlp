import os
import sys
import json
import tensorflow as tf
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("\\hlp\\")])
import hlp.stt.utils.pre_treat as pre_treat


def main():
    parser = ArgumentParser(description='transformer tts V1.0.0')
    parser.add_argument('--config_file', default='', type=str, required=False, help='配置文件路径，为空则默认命令行，不为空则使用配置文件参数')
    parser.add_argument('--act', default='pre_treat', type=str, required=False, help='执行类型')

    parser.add_argument('--max_time_step', default=1100, type=int, required=False, help='最大音频补齐长度')
    parser.add_argument('--audio_feature_type', default='mfcc', type=str, required=False, help='音频处理方式')
    parser.add_argument('--train_data_dir', default='\\data\\data_thchs30\\data\\', type=str, required=False,
                        help='训练数据集存放目录路径')
    parser.add_argument('--train_file', default='\\data\\data_thchs30\\processed_train_file.txt', type=str,
                        required=False, help='整理后的音频句子对保存路径')
    parser.add_argument('--spectrum_data_dir', default='\\data\\data_thchs30\\feature\\', type=str,
                        required=False, help='整理后的音频句子对保存路径')

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    # 注意了，有关路径的参数，以las目录下为基准配置，只要
    # las目录名未更改，任意移动位置不影响使用
    work_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("\\las")]
    execute_type = options['act']

    # optimizer = tf.keras.optimizers.Adam(lr=options['lr'])
    # ckpt_manager = load_checkpoint(encoder=tts_encoder, decoder=tts_decoder,
    #                                checkpoint_dir=work_path + options['checkpoint_dir'],
    #                                execute_type=options['act'], checkpoint_save_size=options['checkpoint_save_size'])

    if execute_type == 'train':
        print("待完善")
    elif execute_type == 'generate':
        print("待完善")
    elif execute_type == 'pre_treat':
        pre_treat.preprocess_thchs30_speech_raw_data(data_path=work_path + options['train_data_dir'],
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
