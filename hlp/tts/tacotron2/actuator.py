import os
import sys
import json
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("\\hlp\\")])
import hlp.tts.utils.pre_treat as pre_treat
import hlp.tts.tacotron2.module as module


def main():
    parser = ArgumentParser(description='%tacotron2 tts')
    parser.add_argument('--config_file', default='', type=str, required=False, help='配置文件路径，为空则默认命令行，不为空则使用配置文件参数')
    parser.add_argument('--type', default='pre_treat', type=str, required=False, help='执行类型')

    parser.add_argument('--metadata_file', default='\\data\\LJSpeech-1.1\\metadata.csv', type=str, required=False,
                        help='原始语音数据的metadata文件路径')
    parser.add_argument('--audio_dir', default='\\data\\LJSpeech-1.1\\wavs\\', type=str, required=False,
                        help='原始语音数据的transcripts文件路径')
    parser.add_argument('--save_file', default='\\data\\audio_sentence_pairs.txt', type=str, required=False,
                        help='原始语音数据的transcripts文件路径')
    parser.add_argument('--cmu_dict_file', default='\\data\\cmudict-0.7b', type=str, required=False,
                        help='cmu音素字典路径')
    parser.add_argument('--spectrum_data_dir', default='\\data\\feature\\', type=str, required=False,
                        help='目标数据存放路径')
    parser.add_argument('--max_length', default=1000, type=int, required=False, help='最大序列长度')
    parser.add_argument('--pre_emphasis', default=0.97, type=float, required=False, help='预加重')
    parser.add_argument('--n_fft', default=2048, type=int, required=False, help='FFT窗口大小')
    parser.add_argument('--n_mels', default=80, type=int, required=False, help='梅尔带数')
    parser.add_argument('--hop_length', default=275, type=int, required=False, help='帧移')
    parser.add_argument('--win_length', default=1102, type=int, required=False, help='窗长')
    parser.add_argument('--max_db', default=100, type=int, required=False, help='峰值分贝值')
    parser.add_argument('--ref_db', default=20, type=int, required=False, help='参考分贝值')
    parser.add_argument('--top_db', default=15, type=int, required=False, help='峰值以下的阈值分贝值')

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    # 注意了，有关路径的参数，以tacotron2目录下为基准配置，只要
    # tacotron2目录名未更改，任意移动位置不影响使用
    work_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("\\tacotron2")]
    execute_type = options['type']

    if execute_type == 'train':
        module.train()
    elif execute_type == 'generate':
        print('待完善')
    elif execute_type == 'pre_treat':
        pre_treat.preprocess_lj_speech_raw_data(metadata_path=work_path + options['metadata_file'],
                                                audio_dir=work_path + options['audio_dir'],
                                                save_path=work_path + options['save_file'],
                                                cmu_dict_path=work_path + options['cmu_dict_file'],
                                                spectrum_data_dir=work_path + options['spectrum_data_dir'],
                                                tokenized_type="phoneme", max_length=options['max_length'],
                                                pre_emphasis=options['pre_emphasis'], n_fft=options['n_fft'],
                                                n_mels=options['n_mels'], hop_length=options['hop_length'],
                                                win_length=options['win_length'], max_db=options['max_db'],
                                                ref_db=options['ref_db'], top_db=options['top_db'])
    else:
        parser.error(msg='')


if __name__ == '__main__':
    """
    Tacotron2入口：指令需要附带运行参数
    cmd：python actuator.py --type [执行模式]
    执行类别：pre_treat/train/evaluate/generate，默认pre_treat模式
    其他参数参见main方法

    generate模式下运行时，输入ESC即退出合成
    """
    main()
