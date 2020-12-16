import os
import sys
import json
from argparse import ArgumentParser
sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("\\hlp\\")])
import hlp.tts.utils.pre_treat as pre_treat
from hlp.tts.utils.utils import load_checkpoint

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
    parser.add_argument('--metadata_file', default='\\data\\LJSpeech-1.1\\metadata.csv', type=str, required=False,
                        help='原始语音数据的metadata文件路径')
    parser.add_argument('--audio_dir', default='\\data\\LJSpeech-1.1\\wavs\\', type=str, required=False,
                        help='原始语音数据的保存目录')
    parser.add_argument('--train_file', default='\\data\\LJSpeech-1.1\\processed_train_file.txt', type=str, required=False,
                        help='整理后的音频句子对保存路径')
    parser.add_argument('--cmu_dict_file', default='\\data\\cmudict-0.7b', type=str, required=False,
                        help='cmu音素字典路径')
    parser.add_argument('--spectrum_data_dir', default='\\data\\LJSpeech-1.1\\feature\\', type=str, required=False,
                        help='目标数据存放路径')
    parser.add_argument('--tokenized_type', default="phoneme", type=str, required=False, help='分词类型')
    parser.add_argument('--max_mel_length', default=1000, type=int, required=False, help='最大序列长度')
    parser.add_argument('--max_sentence_length', default=100, type=int, required=False, help='最大句子序列长度')
    parser.add_argument('--pre_emphasis', default=0.97, type=float, required=False, help='预加重')
    parser.add_argument('--n_fft', default=2048, type=int, required=False, help='FFT窗口大小')
    parser.add_argument('--n_mels', default=80, type=int, required=False, help='梅尔带数')
    parser.add_argument('--hop_length', default=275, type=int, required=False, help='帧移')
    parser.add_argument('--win_length', default=1102, type=int, required=False, help='窗长')
    parser.add_argument('--max_db', default=100, type=int, required=False, help='峰值分贝值')
    parser.add_argument('--ref_db', default=20, type=int, required=False, help='参考分贝值')
    parser.add_argument('--top_db', default=15, type=int, required=False, help='峰值以下的阈值分贝值')
    parser.add_argument('--vocab_size', default=1000, type=int, required=False, help='词汇量大小')
    parser.add_argument('--embedding_dim', default=512, type=int, required=False, help='嵌入层维度')
    parser.add_argument('--num_layers', default=2, type=int, required=False, help='encoder和decoder的layer层数')
    parser.add_argument('--units', default=1024, type=int, required=False, help='单元数')
    parser.add_argument('--num_heads', default=8, type=int, required=False, help='注意头数')
    parser.add_argument('--dropout', default=0.1, type=float, required=False,
                        help='encoder的dropout采样率')

    parser.add_argument('--encoder_pre_net_filters', default=512, type=int, required=False,
                        help='encoder PreNet的卷积输出空间维度')
    parser.add_argument('--encoder_pre_net_kernel_size', default=512, type=int, required=False,
                        help='encoder PreNet的卷积核大小')
    parser.add_argument('--encoder_pre_net_dropout', default=0.2, type=float, required=False,
                        help='encoder PreNet的dropout采样率')
    parser.add_argument('--encoder_pre_net_activation', default='relu', type=str, required=False,
                        help='encoder PreNet的激活函数')
    parser.add_argument('--encoder_pre_net_conv_num', default=3, type=int, required=False,
                        help='encoder PreNet的卷积层数')
    parser.add_argument('--encoder_layer_dropout_rate', default=0.1, type=float, required=False,
                        help='encoder layer的dropout采样率')
    parser.add_argument('--decoder_pre_net_units', default=256, type=int, required=False,
                        help='decoder PreNet的单元数')
    parser.add_argument('--decoder_pre_net_dropout_rate', default=0.1, type=float, required=False,
                        help='decoder PreNet的dropout采样率')
    parser.add_argument('--decoder_pre_net_layers_num', default=2, type=int, required=False,
                        help='decoder PreNet的全连接层数')

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    # 注意了，有关路径的参数，以tacotron2目录下为基准配置，只要
    # tacotron2目录名未更改，任意移动位置不影响使用
    work_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("\\transformer")]
    execute_type = options['act']

    if execute_type == 'train':
        print("待完善")
    elif execute_type == 'generate':
        print("待完善")
        # print("Agent: 你好！结束合成请输入ESC。")
        # while True:
        #     req = input("Sentence: ")
        #     if req == "ESC":
        #         print("Agent: 再见！")
        #         exit(0)
    elif execute_type == 'pre_treat':
        pre_treat.preprocess_lj_speech_raw_data(metadata_path=work_path + options['metadata_file'],
                                                audio_dir=work_path + options['audio_dir'],
                                                dataset_infos_file=work_path + options['train_file'],
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
