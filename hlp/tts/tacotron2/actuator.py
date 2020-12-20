import os
import sys
import json
from argparse import ArgumentParser
import tensorflow as tf
sys.path.append(os.path.abspath(__file__)[:os.path.abspath(__file__).rfind("\\hlp\\")])
import hlp.tts.utils.pre_treat as pre_treat
import hlp.tts.tacotron2.module as module
from hlp.tts.tacotron2.model import Tacotron2
from hlp.tts.tacotron2.module import load_checkpoint


def main():
    parser = ArgumentParser(description='%tacotron2 tts')
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
    parser.add_argument('--checkpoint_dir', default='\\data\\checkpoints\\tacotron2', type=str, required=False,
                        help='检查点保存路径')
    parser.add_argument('--wave_save_dir', default='\\data\\LJSpeech-1.1\\generate\\tacotron2', type=str,
                        required=False,
                        help='合成的音频保存路径')
    parser.add_argument('--valid_data_path', default="", type=str, required=False, help='验证集路径')
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
    parser.add_argument('--epochs', default=4, type=int, required=False, help='训练周期')
    parser.add_argument('--vocab_size', default=500, type=int, required=False, help='词汇大小')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='batch大小')
    parser.add_argument('--buffer_size', default=20000, type=int, required=False, help='dataset缓冲区大小')
    parser.add_argument('--checkpoint_save_size', default=2, type=int, required=False, help='训练中最多保存checkpoint数量')
    parser.add_argument('--checkpoint_save_freq', default=2, type=int, required=False, help='检查点保存频率')
    parser.add_argument('--tokenized_type', default="phoneme", type=str, required=False, help='分词类型')
    parser.add_argument('--valid_data_split', default=0.2, type=float, required=False, help='从训练数据划分验证数据的比例')
    parser.add_argument('--max_train_data_size', default=0, type=int, required=False, help='从训练集读取最大数据量，0即全部数据')
    parser.add_argument('--max_valid_data_size', default=0, type=int, required=False, help='从验证集集读取最大数据量，0即全部数据')
    parser.add_argument('--lr', default=0.0001, type=float, required=False, help='学习率')
    parser.add_argument('--sr', default=22050, type=int, required=False, help='采样率')
    parser.add_argument('--n_conv_encoder', default=3, type=int, required=False, help='')
    parser.add_argument('--encoder_conv_filters', default=256, type=int, required=False, help='')
    parser.add_argument('--encoder_conv_kernel_sizes', default=5, type=int, required=False, help='')
    parser.add_argument('--encoder_conv_activation', default="relu", type=str, required=False, help='')
    parser.add_argument('--encoder_conv_dropout_rate', default=0.1, type=float, required=False, help='')
    parser.add_argument('--encoder_lstm_units', default=256, type=int, required=False, help='')
    parser.add_argument('--attention_rnn_dim', default=512, type=int, required=False, help='')
    parser.add_argument('--decoder_lstm_dim', default=512, type=int, required=False, help='')
    parser.add_argument('--decoder_lstm_rate', default=0.1, type=float, required=False, help='')
    parser.add_argument('--initial_hidden_size', default=512, type=int, required=False, help='')
    parser.add_argument('--attention_dim', default=128, type=int, required=False, help='')
    parser.add_argument('--attention_filters', default=32, type=int, required=False, help='')
    parser.add_argument('--attention_kernel', default=31, type=int, required=False, help='')
    parser.add_argument('--n_prenet_layers', default=2, type=int, required=False, help='')
    parser.add_argument('--prenet_units', default=256, type=int, required=False, help='')
    parser.add_argument('--prenet_dropout_rate', default=0.1, type=float, required=False, help='')
    parser.add_argument('--gate_threshold', default=0.5, type=float, required=False, help='')
    parser.add_argument('--n_conv_postnet', default=3, type=int, required=False, help='')
    parser.add_argument('--postnet_conv_filters', default=256, type=int, required=False, help='')
    parser.add_argument('--postnet_conv_kernel_sizes', default=5, type=int, required=False, help='')
    parser.add_argument('--postnet_dropout_rate', default=0.1, type=float, required=False, help='')
    parser.add_argument('--postnet_conv_activation', default="tanh", type=str, required=False, help='')
    parser.add_argument('--embedding_hidden_size', default=64, type=int, required=False, help='')
    parser.add_argument('--n_iter', default=100, type=int, required=False, help='')

    options = parser.parse_args().__dict__
    if options['config_file'] != '':
        with open(options['config_file'], 'r', encoding='utf-8') as config_file:
            options = json.load(config_file)

    # 注意了，有关路径的参数，以tacotron2目录下为基准配置，只要
    # tacotron2目录名未更改，任意移动位置不影响使用
    work_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("\\tacotron2")]
    execute_type = options['act']

    model = Tacotron2(vocab_size=options['vocab_size'], encoder_conv_filters=options['encoder_conv_filters'],
                      encoder_conv_kernel_sizes=options['encoder_conv_kernel_sizes'],
                      encoder_conv_activation=options['encoder_conv_activation'],
                      encoder_lstm_units=options['encoder_lstm_units'],
                      encoder_conv_dropout_rate=options['encoder_conv_dropout_rate'],
                      embedding_hidden_size=options['embedding_hidden_size'],
                      n_conv_encoder=options['n_conv_encoder'], attention_dim=options['attention_dim'],
                      attention_filters=options['attention_filters'], attention_kernel=options['attention_kernel'],
                      n_prenet_layers=options['n_prenet_layers'], prenet_dropout_rate=options['prenet_dropout_rate'],
                      n_conv_postnet=options['n_conv_postnet'], postnet_conv_filters=options['postnet_conv_filters'],
                      postnet_conv_kernel_sizes=options['postnet_conv_kernel_sizes'],
                      postnet_dropout_rate=options['postnet_dropout_rate'], prenet_units=options['prenet_units'],
                      postnet_conv_activation=options['postnet_conv_activation'], n_mels=options['n_mels'],
                      attention_rnn_dim=options['attention_rnn_dim'], decoder_lstm_dim=options['decoder_lstm_dim'],
                      gate_threshold=options['gate_threshold'], max_input_length=options['max_mel_length'],
                      initial_hidden_size=options['initial_hidden_size'],
                      decoder_lstm_rate=options['decoder_lstm_rate'])
    optimizer = tf.keras.optimizers.Adam(lr=options['lr'])
    ckpt_manager = load_checkpoint(model=model, checkpoint_dir=work_path + options['checkpoint_dir'],
                                   execute_type=options['act'], checkpoint_save_size=options['checkpoint_save_size'])

    if execute_type == 'train':
        module.train(epochs=options['epochs'], train_data_path=work_path + options['train_file'],
                     max_len=options['max_sentence_length'], vocab_size=options['vocab_size'],
                     batch_size=options['batch_size'], buffer_size=options['buffer_size'],
                     checkpoint=ckpt_manager, model=model, optimizer=optimizer, valid_data_path="",
                     tokenized_type=options['tokenized_type'], dict_path=work_path + options['dict_path'],
                     valid_data_split=options['valid_data_split'], max_train_data_size=options['max_train_data_size'],
                     max_valid_data_size=options['max_valid_data_size'],
                     checkpoint_save_freq=options['checkpoint_save_freq'])
    elif execute_type == 'evaluate':
        module.evaluate(model=model, data_path=work_path + options['train_file'],
                        max_len=options['max_sentence_length'], vocab_size=options['vocab_size'],
                        max_train_data_size=options['max_train_data_size'], batch_size=options['batch_size'],
                        buffer_size=options['buffer_size'], tokenized_type="phoneme")
    elif execute_type == 'generate':
        module.generate(model=model, max_db=options['max_db'], ref_db=options['ref_db'],
                        sr=options['sr'], max_len=options['max_sentence_length'],
                        wave_save_dir=work_path + options['wave_save_dir'], n_fft=options['n_fft'],
                        n_mels=options['n_mels'], pre_emphasis=options['pre_emphasis'],
                        n_iter=options['n_iter'], hop_length=options['hop_length'],
                        win_length=options['win_length'], dict_path=work_path + options['dict_path'],
                        tokenized_type=options['tokenized_type'], cmu_dict_path=work_path + options['cmu_dict_file'])
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


if __name__ == '__main__':
    """
    Tacotron2入口：指令需要附带运行参数
    cmd：python actuator.py --act [执行模式]
    执行类别：pre_treat/train/evaluate/generate，默认pre_treat模式
    其他参数参见main方法

    generate模式下运行时，输入ESC即退出合成
    """
    main()
