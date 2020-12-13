import os
import json
from argparse import ArgumentParser
import hlp.tts.utils.pre_treat as pre_treat

if __name__ == '__main__':
    """
    Transformer TTS入口：指令需要附带运行参数
    cmd：python transformer_launch.py -t/--type [执行模式]
    执行类别：pre_treat/train/generate

    generate模式下运行时，输入ESC即退出语音合成
    """
    parser = ArgumentParser(version='transformer tts V1.0.0')
    parser.add_argument('--config_file', default='', type=str, required=False, help='配置文件路径，为空则默认命令行，不为空则使用配置文件参数')
    parser.add_argument('--type', default='pre_treat', type=str, required=False, help='执行类型')
    parser.add_argument('--metadata_file', default='\\data\\LJSpeech-1.1\\metadata.csv', type=str, required=False,
                        help='原始语音数据的metadata文件路径')

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    # 注意了，有关路径的参数，以tacotron2目录下为基准配置，只要
    # tacotron2目录名未更改，任意移动位置不影响使用
    work_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("\\tacotron2")]
    execute_type = options['type']

    if options.type == 'train':
        print("待完善")
    elif options.type == 'chat':
        print("待完善")
        # print("Agent: 你好！结束合成请输入ESC。")
        # while True:
        #     req = input("Sentence: ")
        #     if req == "ESC":
        #         print("Agent: 再见！")
        #         exit(0)
    elif options.type == 'pre_treat':
        pre_treat.preprocess_lj_speech_raw_data(metadata_path=work_path + options['metadata_file'],
                                                audio_dir=work_path + options['audio_dir'],
                                                save_path=work_path + options['train_file'],
                                                cmu_dict_path=work_path + options['cmu_dict_file'],
                                                spectrum_data_dir=work_path + options['spectrum_data_dir'],
                                                tokenized_type=options['tokenized_type'],
                                                max_length=options['max_mel_length'],
                                                pre_emphasis=options['pre_emphasis'], n_fft=options['n_fft'],
                                                n_mels=options['n_mels'], hop_length=options['hop_length'],
                                                win_length=options['win_length'], max_db=options['max_db'],
                                                ref_db=options['ref_db'], top_db=options['top_db'])
    else:
        parser.error(msg='')
