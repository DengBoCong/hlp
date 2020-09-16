import os
import sys
from pathlib import Path
import tensorflow as tf
from optparse import OptionParser
import config.get_config as _config
import model.seq2seq.model as seq2seq_model
import model.seq2seq.trainer as seq2seq_train
import model.seq2seq.predict as seq2seq_predict
from common.pre_treat import preprocess_raw_data

if __name__ == '__main__':
    """
    Seq2Seq入口：指令需要附带运行参数
    cmd：python seq2seq.py -t/--type [执行模式]
    执行类别：pre_treat/train/chat

    chat模式下运行时，输入exit即退出对话
    """
    parser = OptionParser(version='%seq2seq chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, pre_treat/train/chat")
    (options, args) = parser.parse_args()
    if options.type == 'train':
        seq2seq_train.train()
    elif options.type == 'chat':
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
    elif options.type == 'pre_treat':
        preprocess_raw_data()
    else:
        print('Error:不存在', sys.argv[2], '模式!')
