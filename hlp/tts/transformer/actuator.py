import os
import sys
import json
import tensorflow as tf
from argparse import ArgumentParser
sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("\\hlp\\")])
import hlp.tts.utils.pre_treat as pre_treat
import hlp.tts.transformer.module as module
from hlp.tts.transformer.model import encoder
from hlp.tts.transformer.model import decoder
from hlp.tts.transformer.module import load_checkpoint


def main():
    parser = ArgumentParser(description='transformer tts V1.0.0')
    parser.add_argument('--config_file', default='', type=str, required=False, help='配置文件路径，为空则默认命令行，不为空则使用配置文件参数')
    parser.add_argument('--act', default='pre_treat', type=str, required=False, help='执行类型')
    parser.add_argument('--metadata_file', default='\\data\\LJSpeech-1.1\\metadata.csv', type=str, required=False,
                        help='原始语音数据的metadata文件路径')
    parser.add_argument('--audio_dir', default='\\data\\LJSpeech-1.1\\wavs\\', type=str, required=False,
                        help='原始语音数据的保存目录')
    parser.add_argument('--train_file', default='\\data\\LJSpeech-1.1\\processed_train_file.txt', type=str,
                        required=False,
                        help='整理后的音频句子对保存路径')
    parser.add_argument('--cmu_dict_file', default='\\data\\cmudict-0.7b', type=str, required=False,
                        help='cmu音素字典路径')
    parser.add_argument('--spectrum_data_dir', default='\\data\\LJSpeech-1.1\\feature\\', type=str, required=False,
                        help='目标数据存放路径')
    parser.add_argument('--dict_path', default='\\data\\LJSpeech-1.1\\tacotron2_dict.json', type=str, required=False,
                        help='字典存放路径')
    parser.add_argument('--checkpoint_dir', default='\\data\\checkpoints\\transformer', type=str, required=False,
                        help='检查点保存路径')
    parser.add_argument('--wave_save_dir', default='\\data\\LJSpeech-1.1\\generate\\transformer', type=str,
                        required=False, help='合成的音频保存路径')
    parser.add_argument('--tokenized_type', default="phoneme", type=str, required=False, help='分词类型')
    parser.add_argument('--max_mel_length', default=700, type=int, required=False, help='最大序列长度')
    parser.add_argument('--max_sentence_length', default=100, type=int, required=False, help='最大句子序列长度')
    parser.add_argument('--pre_emphasis', default=0.97, type=float, required=False, help='预加重')
    parser.add_argument('--n_fft', default=2048, type=int, required=False, help='FFT窗口大小')
    parser.add_argument('--num_mel', default=80, type=int, required=False, help='梅尔带数')
    parser.add_argument('--hop_length', default=275, type=int, required=False, help='帧移')
    parser.add_argument('--win_length', default=1102, type=int, required=False, help='窗长')
    parser.add_argument('--max_db', default=100, type=int, required=False, help='峰值分贝值')
    parser.add_argument('--ref_db', default=20, type=int, required=False, help='参考分贝值')
    parser.add_argument('--top_db', default=15, type=int, required=False, help='峰值以下的阈值分贝值')
    parser.add_argument('--n_iter', default=100, type=int, required=False, help='')
    parser.add_argument('--vocab_size', default=1000, type=int, required=False, help='词汇量大小')
    parser.add_argument('--embedding_dim', default=256, type=int, required=False, help='嵌入层维度')
    parser.add_argument('--num_layers', default=6, type=int, required=False, help='encoder和decoder的layer层数')
    parser.add_argument('--encoder_units', default=256, type=int, required=False, help='单元数')
    parser.add_argument('--decoder_units', default=256, type=int, required=False, help='单元数')
    parser.add_argument('--num_heads', default=8, type=int, required=False, help='注意头数')
    parser.add_argument('--dropout', default=0.1, type=float, required=False,
                        help='encoder的dropout采样率')
    parser.add_argument('--lr', default=0.0001, type=float, required=False, help='学习率')
    parser.add_argument('--sr', default=22050, type=int, required=False, help='采样率')
    parser.add_argument('--epochs', default=4, type=int, required=False, help='训练轮次')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='batch大小')
    parser.add_argument('--buffer_size', default=20000, type=int, required=False, help='dataset缓冲区大小')
    parser.add_argument('--checkpoint_save_freq', default=2, type=int, required=False, help='检查点保存频率')
    parser.add_argument('--valid_data_split', default=0.2, type=float, required=False, help='从训练数据划分验证数据的比例')
    parser.add_argument('--max_train_data_size', default=0, type=int, required=False, help='从训练集读取最大数据量，0即全部数据')
    parser.add_argument('--max_valid_data_size', default=0, type=int, required=False, help='从验证集集读取最大数据量，0即全部数据')
    parser.add_argument('--checkpoint_save_size', default=2, type=int, required=False, help='训练中最多保存checkpoint数量')
    parser.add_argument('--encoder_pre_net_filters', default=256, type=int, required=False,
                        help='encoder PreNet的卷积输出空间维度')
    parser.add_argument('--encoder_pre_net_kernel_size', default=256, type=int, required=False,
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
    parser.add_argument('--post_net_conv_num', default=5, type=int, required=False, help='PostNet的卷积层数')
    parser.add_argument('--post_net_filters', default=256, type=int, required=False, help='PostNet的卷积输出空间维数')
    parser.add_argument('--post_net_kernel_sizes', default=5, type=int, required=False, help='PostNet的卷积核大小')
    parser.add_argument('--post_net_dropout', default=0.1, type=float, required=False, help='PostNet的dropout采样率')
    parser.add_argument('--post_net_activation', default='tanh', type=str, required=False, help='PostNet的卷积激活函数')

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    # 注意了，有关路径的参数，以transformer目录下为基准配置，只要
    # transformer目录名未更改，任意移动位置不影响使用
    work_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("\\transformer")]
    execute_type = options['act']

    tts_encoder = encoder(vocab_size=options['vocab_size'], embedding_dim=options['embedding_dim'],
                          encoder_pre_net_conv_num=options['encoder_pre_net_conv_num'],
                          num_layers=options['num_layers'], encoder_pre_net_filters=options['encoder_pre_net_filters'],
                          encoder_pre_net_kernel_size=options['encoder_pre_net_kernel_size'],
                          encoder_pre_net_activation=options['encoder_pre_net_activation'],
                          encoder_units=options['encoder_units'], num_heads=options['num_heads'],
                          encoder_layer_dropout_rate=options['encoder_layer_dropout_rate'],
                          encoder_pre_net_dropout=options['encoder_pre_net_dropout'], dropout=options['dropout'])

    tts_decoder = decoder(vocab_size=options['vocab_size'], embedding_dim=options['embedding_dim'],
                          num_layers=options['num_layers'], decoder_units=options['decoder_units'],
                          num_heads=options['num_heads'], post_net_filters=options['post_net_filters'],
                          num_mel=options['num_mel'], post_net_activation=options['post_net_activation'],
                          decoder_pre_net_layers_num=options['decoder_pre_net_layers_num'],
                          post_net_conv_num=options['post_net_conv_num'], dropout=options['dropout'],
                          post_net_kernel_sizes=options['post_net_kernel_sizes'],
                          decoder_pre_net_dropout_rate=options['decoder_pre_net_dropout_rate'],
                          post_net_dropout=options['post_net_dropout'])
    optimizer = tf.keras.optimizers.Adam(lr=options['lr'])
    ckpt_manager = load_checkpoint(encoder=tts_encoder, decoder=tts_decoder,
                                   checkpoint_dir=work_path + options['checkpoint_dir'],
                                   execute_type=options['act'], checkpoint_save_size=options['checkpoint_save_size'])

    if execute_type == 'train':
        module.train(encoder=tts_encoder, decoder=tts_decoder, optimizer=optimizer,
                     epochs=options['epochs'], checkpoint=ckpt_manager, batch_size=options['batch_size'],
                     train_data_path=work_path + options['train_file'], valid_data_path="",
                     max_len=options['max_sentence_length'], vocab_size=options['vocab_size'],
                     buffer_size=options['buffer_size'], checkpoint_save_freq=options['checkpoint_save_freq'],
                     tokenized_type=options['tokenized_type'], dict_path=work_path + options['dict_path'],
                     valid_data_split=options['valid_data_split'], max_valid_data_size=options['max_valid_data_size'],
                     max_train_data_size=options['max_train_data_size'], num_mel=options['num_mel'])
    elif execute_type == 'generate':
        module.generate(encoder=tts_encoder, decoder=tts_decoder, num_mel=options['num_mel'],
                        wave_save_dir=work_path + options['wave_save_dir'], max_mel_length=options['max_mel_length'],
                        cmu_dict_path=work_path + options['cmu_dict_file'], tokenized_type=options['tokenized_type'],
                        dict_path=work_path + options['dict_path'], max_len=options['max_sentence_length'],
                        max_db=options['max_db'], ref_db=options['ref_db'], sr=options['sr'], n_fft=options['n_fft'],
                        pre_emphasis=options['pre_emphasis'], n_iter=options['n_iter'],
                        hop_length=options['hop_length'], win_length=options['win_length'])
    elif execute_type == 'pre_treat':
        pre_treat.preprocess_lj_speech_raw_data(metadata_path=work_path + options['metadata_file'],
                                                audio_dir=work_path + options['audio_dir'],
                                                dataset_infos_file=work_path + options['train_file'],
                                                cmu_dict_path=work_path + options['cmu_dict_file'],
                                                spectrum_data_dir=work_path + options['spectrum_data_dir'],
                                                tokenized_type=options['tokenized_type'],
                                                max_length=options['max_mel_length'],
                                                pre_emphasis=options['pre_emphasis'], n_fft=options['n_fft'],
                                                n_mels=options['num_mel'], hop_length=options['hop_length'],
                                                win_length=options['win_length'], max_db=options['max_db'],
                                                ref_db=options['ref_db'], top_db=options['top_db'])
    else:
        parser.error(msg='')


if __name__ == '__main__':
    """
    Transformer TTS入口：指令需要附带运行参数
    cmd：python actuator.py --act [执行模式]
    执行类别：pre_treat/train/generate

    generate模式下运行时，输入ESC即退出语音合成
    """
    main()
