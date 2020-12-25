import os
import sys
import json
import tensorflow as tf
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("\\hlp\\")])
from hlp.stt.utils.pre_treat import dispatch_pre_treat_func
from hlp.stt.transformer.module import train
from hlp.stt.transformer.module import recognize
from hlp.stt.transformer.module import load_checkpoint
from hlp.stt.transformer.model import encoder
from hlp.stt.transformer.model import decoder

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
    parser.add_argument('--encoder_vocab_size', default=10000, type=int, required=False, help='词汇量大小')
    parser.add_argument('--decoder_vocab_size', default=1000, type=int, required=False, help='词汇量大小')
    parser.add_argument('--feature_dim', default=80, type=int, required=False, help='处理后音频的特征维度')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='batch大小')
    parser.add_argument('--buffer_size', default=20000, type=int, required=False, help='dataset缓冲区大小')
    parser.add_argument('--beam_size', default=3, type=int, required=False, help='beam_size')
    parser.add_argument('--checkpoint_save_freq', default=2, type=int, required=False, help='检查点保存频率')
    parser.add_argument('--embedding_dim', default=256, type=int, required=False, help='嵌入层维度')
    parser.add_argument('--encoder_units', default=256, type=int, required=False, help='单元数')
    parser.add_argument('--decoder_units', default=256, type=int, required=False, help='单元数')
    parser.add_argument('--num_heads', default=8, type=int, required=False, help='注意头数')
    parser.add_argument('--dropout', default=0.1, type=float, required=False, help='encoder的dropout采样率')
    parser.add_argument('--num_layers', default=2, type=int, required=False, help='encoder和decoder的layer层数')
    parser.add_argument('--max_sentence_length', default=100, type=int, required=False, help='最大句子序列长度')
    parser.add_argument('--valid_data_split', default=0.1, type=float, required=False, help='从训练数据划分验证数据的比例')
    parser.add_argument('--max_train_data_size', default=20, type=int, required=False, help='从训练集读取最大数据量，0即全部数据')
    parser.add_argument('--max_valid_data_size', default=0, type=int, required=False, help='从验证集集读取最大数据量，0即全部数据')
    parser.add_argument('--checkpoint_save_size', default=2, type=int, required=False, help='训练中最多保存checkpoint数量')
    parser.add_argument('--dataset_type', default='thchs30', type=str, required=False, help='数据集类型')
    parser.add_argument('--audio_feature_type', default='fbank', type=str, required=False, help='音频处理方式')
    parser.add_argument('--max_time_step', default=1100, type=int, required=False, help='最大音频补齐长度')
    parser.add_argument('--train_data_dir', default='\\data\\data_thchs30\\data\\', type=str, required=False,
                        help='训练数据集存放目录路径')
    parser.add_argument('--train_file', default='\\data\\data_thchs30\\processed_train_file.txt', type=str,
                        required=False, help='整理后的音频句子对保存路径')
    parser.add_argument('--spectrum_data_dir', default='\\data\\data_thchs30\\feature\\', type=str,
                        required=False, help='整理后的音频句子对保存路径')
    parser.add_argument('--dict_path', default='\\data\\data_thchs30\\transformer_dict.json', type=str, required=False,
                        help='字典存放路径')
    parser.add_argument('--checkpoint_dir', default='\\data\\checkpoints\\transformer', type=str, required=False,
                        help='检查点保存路径')

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    # 注意了，有关路径的参数，以tacotron2目录下为基准配置，只要
    # tacotron2目录名未更改，任意移动位置不影响使用
    work_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("\\transformer")]
    execute_type = options['act']

    encoder = encoder(vocab_size=options['encoder_vocab_size'], embedding_dim=options['embedding_dim'],
                      feature_dim=options['feature_dim'], num_layers=options['num_layers'],
                      encoder_units=options['encoder_units'],
                      num_heads=options['num_heads'], dropout=options['dropout'])
    decoder = decoder(vocab_size=options['decoder_vocab_size'], embedding_dim=options['embedding_dim'],
                      num_layers=options['num_layers'], decoder_units=options['decoder_units'],
                      num_heads=options['num_heads'], dropout=options['dropout'])

    optimizer = tf.keras.optimizers.Adam(lr=options['lr'])
    ckpt_manager = load_checkpoint(encoder=encoder, decoder=decoder,
                                   checkpoint_dir=work_path + options['checkpoint_dir'],
                                   execute_type=options['act'], checkpoint_save_size=options['checkpoint_save_size'])

    if execute_type == 'train':
        train(epochs=options['epochs'], train_data_path=work_path + options['train_file'],
              max_len=options['max_sentence_length'], vocab_size=options['decoder_vocab_size'],
              batch_size=options['batch_size'], buffer_size=options['buffer_size'],
              checkpoint_save_freq=options['checkpoint_save_freq'], checkpoint=ckpt_manager,
              optimizer=optimizer, dict_path=work_path + options['dict_path'],
              valid_data_split=options['valid_data_split'], valid_data_path="",
              max_train_data_size=options['max_train_data_size'], encoder=encoder,
              max_valid_data_size=options['max_valid_data_size'], decoder=decoder)
    elif execute_type == 'recognize':
        recognize(encoder=encoder, decoder=decoder, beam_size=options['beam_size'],
                  audio_feature_type=options['audio_feature_type'], max_length=options['max_time_step'],
                  max_sentence_length=options['max_sentence_length'], dict_path=work_path + options['dict_path'])
    elif execute_type == 'pre_treat':
        dispatch_pre_treat_func(func_type=options['dataset_type'],
                                data_path=work_path + options['train_data_dir'],
                                dataset_infos_file=work_path + options['train_file'],
                                max_length=options['max_time_step'], transcript_row=0,
                                spectrum_data_dir=work_path + options['spectrum_data_dir'],
                                audio_feature_type=options['audio_feature_type'])
    else:
        parser.error(msg='')
