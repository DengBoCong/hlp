from model import bleu,trainer,translator
from optparse import OptionParser
from config import get_config as _config
from pathlib import Path
import os
import tensorflow as tf

'''

程序入口

共有三种模式：
- train: 使用 ./data 文件夹下的指定文件(默认 en-ch.txt)进行训练
- bleu : 使用 ./data 文件夹下的指定文件(默认 en-ch_evaluate.txt)进行BLEU指标计算
- translate : 对指定输入句子进行翻译，输入exit退出

cmd: python transformer.py -t/--type [执行模式]

'''


def main():
    parser = OptionParser(version='%prog V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="train",
                      help="execute type: train/bleu/translate")
    (options, args) = parser.parse_args()
    if options.type == 'train':
        trainer.train()
    elif options.type == 'bleu' or options.type == 'translate':
        # 检查是否有checkpoint
        checkpoint_dir = _config.checkpoint_path
        is_exist = Path(checkpoint_dir)
        is_exist = Path(checkpoint_dir)
        if not is_exist.exists():
            os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt = tf.io.gfile.listdir(checkpoint_dir)
        if ckpt:
            if options.type == 'translate':
                while (True):
                    sentence = input('请输入要翻译的句子 :')
                    if sentence == 'exit':
                        break
                    else:
                        print('translate    :', translator.translate(sentence))
            else:
                bleu.calc_bleu()
        else:
            print('请先训练再进行测试体验...')


if __name__ == '__main__':
    main()
