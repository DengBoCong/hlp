import os
import sys
import json
import tensorflow as tf
from argparse import ArgumentParser
sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("\\hlp\\")])

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

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    # 注意了，有关路径的参数，以tacotron2目录下为基准配置，只要
    # tacotron2目录名未更改，任意移动位置不影响使用
    work_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("\\transformer")]
    execute_type = options['act']


    optimizer = tf.keras.optimizers.Adam(lr=options['lr'])
    # ckpt_manager = load_checkpoint(encoder=tts_encoder, decoder=tts_decoder,
    #                                checkpoint_dir=work_path + options['checkpoint_dir'],
    #                                execute_type=options['act'], checkpoint_save_size=options['checkpoint_save_size'])

    if execute_type == 'train':
        print("待完善")
    elif execute_type == 'generate':
        print("待完善")
    elif execute_type == 'pre_treat':
        print("待完善")
    else:
        parser.error(msg='')
