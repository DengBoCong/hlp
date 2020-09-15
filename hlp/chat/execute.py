import sys
import os
from pathlib import Path
import tensorflow as tf
from optparse import OptionParser
import config.get_config as _config
import model.seq2seq.model as seq2seq_model
import model.seq2seq.trainer as seq2seq_train
import model.seq2seq.predict as seq2seq_predict
import model.transformer.model as transformer_model
import model.transformer.trainer as transformer_train
import model.transformer.predict as transformer_predict
from common.pre_treat import preprocess_raw_data

'''
主入口：指令需要附带运行参数
cmd：python execute.py [模型类别] [执行模式]
模型类别：seq2seq/gpt2
执行类别：chat/train

chat模式下运行时，输入exit即退出对话
'''

parser = OptionParser(version="%seq2seq chatbot V1.0")
parser.add_option("-t", "--type", action="store", type="string",
                  dest="type", default="seq2seq",
                  help="model type, seq2seq/transformer/gpt2")
parser.add_option("-k", "--kind", action="store", type="string",
                  dest="kind", default="pre_treat",
                  help="execute type, chat/train/pre_treat")

(options, args) = parser.parse_args()

if __name__ == '__main__':
    if options.type == 'seq2seq':
        if options.kind == 'train':
            seq2seq_train.train()
        elif options.kind == 'chat':
            checkpoint_dir = _config.seq2seq_train_data

            # 这里需要检查一下是否有模型的目录，没有的话就创建，有的话就跳过
            is_exist = Path(checkpoint_dir)
            if not is_exist.exists():
                os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt = tf.io.gfile.listdir(checkpoint_dir)
            if ckpt:
                seq2seq_model.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
                while (True):
                    sentence = input('User:')
                    if sentence == 'exit':
                        break
                    else:
                        print('ChatBot:', seq2seq_predict.predict(sentence, seq2seq_model))
            else:
                print('请先训练再进行测试体验，训练轮数建议一百轮以上!')
        elif options.kind == 'pre_treat':
            preprocess_raw_data()
        else:
            print('Error:不存在', sys.argv[2], '模式!')
    elif options.type == 'transformer':
        if options.kind == 'train':
            transformer_train.train()
        elif options.kind == 'chat':
            checkpoint_dir = _config.transformer_train_data
            is_exist = Path(checkpoint_dir)
            if not is_exist.exists():
                os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt = tf.io.gfile.listdir(checkpoint_dir)
            if ckpt:
                transformer_model.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
                while (True):
                    sentence = input('User:')
                    if sentence == 'exit':
                        break
                    else:
                        print('ChatBot:', transformer_predict.predict(sentence))
            else:
                print('请先训练再进行测试体验，训练轮数建议一百轮以上!')
        elif options.kind == 'pre_treat':
            preprocess_raw_data()
        else:
            print('Error:不存在', sys.argv[2], '模式!')
    elif options.type == 'gpt2':
        print('gpt2模型开发中，敬请期待!')
    else:
        print('Error:不存在', sys.argv[1], '模型类别!')
