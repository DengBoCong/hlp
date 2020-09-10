import sys
import os
from model.Seq2Seq.trainer import train
from model.Seq2Seq.predict import predict
from common.pre_treat import preprocess_raw_data

'''
主入口：指令需要附带运行参数
cmd：python execute.py [模型类别] [执行模式]
模型类别：seq2seq/gpt2
执行类别：chat/train

chat模式下运行时，输入exit即退出对话
'''

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Error:指令执行有误，详细见执行说明!')

    elif sys.argv[1] == 'seq2seq':
        if sys.argv[2] == 'train':
            train()
        elif sys.argv[2] == 'chat':
            while (True):
                sentence = input('User:')
                if sentence == 'exit':
                    break
                else:
                    print('ChatBot:', predict(sentence))
        elif sys.argv[2] == 'pre_treat':
            preprocess_raw_data()
        else:
            print('Error:不存在', sys.argv[2], '模式!')
    else:
        print('Error:不存在', sys.argv[1], '模型类别!')
