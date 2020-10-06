import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model import evaluate,trainer,translator
from optparse import OptionParser
from model import evaluate as eval
from common import preprocess as _pre
from model import network
from config import get_config as _config
'''

程序入口

共有三种模式：
- train: 使用 ./data 文件夹下的指定文件(默认 en-ch.txt)进行训练
- eval : 使用 ./data 文件夹下的指定文件(默认 en-ch_evaluate.txt)对模型进行评价,需对指标类型进行选择
    - bleu 指标
- translate : 对指定输入句子进行翻译，输入exit退出

cmd: python nml.py -t/--type [执行模式]

'''


def main():
    # 配置命令行参数
    parser = OptionParser(version='%prog V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="translate",
                      help="可选择的模式: train/eval/translate")
    (options, args) = parser.parse_args()

    if options.type == 'train':
        # 加载句子
        en, ch = _pre.load_sentences(_config.path_to_train_file, _config.num_sentences)
        # 预处理句子
        en = _pre.preprocess_sentences_en(en, mode=_config.en_tokenize_type)
        ch = _pre.preprocess_sentences_ch(ch, mode=_config.ch_tokenize_type)
        # 生成及保存字典
        tokenizer_en, vocab_size_en = _pre.create_tokenizer(sentences=en, mode=_config.en_tokenize_type
                                             , save_path=_config.en_bpe_tokenizer_path)
        tokenizer_ch, vocab_size_ch = _pre.create_tokenizer(sentences=ch, mode=_config.ch_tokenize_type
                                             , save_path=_config.ch_tokenizer_path)
        print('vocab_size_en:%d' % vocab_size_en)
        print('vocab_size_ch:%d' % vocab_size_ch)
        # 编码句子
        tensor_en, max_sequence_length_en = _pre.encode_sentences(sentences=en, tokenizer=tokenizer_en
                                                                  , mode=_config.en_tokenize_type)
        tensor_ch, max_sequence_length_ch = _pre.encode_sentences(sentences=ch, tokenizer=tokenizer_ch
                                                                  , mode=_config.ch_tokenize_type)

        # 创建模型及相关变量
        optimizer, train_loss, train_accuracy, transformer = network.get_model(vocab_size_en, vocab_size_ch)
        # 开始训练
        trainer.train(tensor_en, tensor_ch, transformer, optimizer, train_loss, train_accuracy)

    elif options.type == 'eval' or options.type == 'translate':
        if_ckpt = _pre.check_point()  # 检测是否有检查点
        if if_ckpt:
            # 加载中英文字典
            tokenizer_en, vocab_size_en = _pre.get_tokenizer(path=_config.en_bpe_tokenizer_path
                                                             , mode=_config.en_tokenize_type)
            tokenizer_ch, vocab_size_ch = _pre.get_tokenizer(path=_config.ch_tokenizer_path
                                                             , mode=_config.ch_tokenize_type)
            print('vocab_size_en:%d' % vocab_size_en)
            print('vocab_size_ch:%d' % vocab_size_ch)
            # 创建模型及相关变量
            optimizer, _, _, transformer = network.get_model(vocab_size_en, vocab_size_ch)
            # 加载检查点
            network.load_checkpoint(transformer, optimizer)
            if options.type == 'eval':
                # 评估模式
                print('-' * 30)
                print('可选择评价指标： 1.bleu指标  0.退出程序')
                eval_mode = input('请输入选择指标的序号：')
                if eval_mode == '1':
                    eval.calc_bleu(_config.path_to_eval_file, transformer, tokenizer_en, tokenizer_ch)
                elif eval_mode == '0':
                    print('感谢您的体验！')
                else:
                    print('请输入正确序号')
            elif options.type == 'translate':
                # 翻译模式
                while True:
                    print('-'*30)
                    print('输入0可退出程序')
                    sentence = input('请输入要翻译的句子 :')
                    if sentence == '0':
                        break
                    else:
                        print('翻译结果:', translator.translate(sentence, transformer, tokenizer_en, tokenizer_ch))
        else:
            print('请先训练才可使用其它功能...')
    else:
        print('未知模式：' + sys.argv[2])
        print(parser.format_help())


if __name__ == '__main__':
    main()