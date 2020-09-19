import os
import sys
from pathlib import Path
import tensorflow as tf
from optparse import OptionParser
import config.get_config as _config
import model.transformer.model as transformer_model
import model.transformer.trainer as transformer_train
import model.transformer.predict as transformer_predict
from common.pre_treat import preprocess_raw_data

if __name__ == '__main__':

    parser = OptionParser(version='%transformer chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, pre_treat/train/chat")
    (options, args) = parser.parse_args()
    if options.type == 'train':
        transformer_train.train()
    elif options.type == 'chat':
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
    elif options.type == 'pre_treat':
        preprocess_raw_data()
    else:
        print('Error:不存在', sys.argv[2], '模式!')
