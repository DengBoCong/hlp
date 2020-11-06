import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model import evaluate, trainer, translator
from common import preprocess as _pre
from model import nmt_model

"""
使用 ./data 文件夹下的指定文件(默认 en-ch.txt)进行训练
"""


def main():
    # 进行训练所需的数据处理
    vocab_size_source, vocab_size_target = _pre.train_preprocess()

    # 创建模型及相关变量
    transformer = nmt_model.get_model(vocab_size_source, vocab_size_target)

    # 开始训练
    trainer.train(transformer)


if __name__ == '__main__':
    main()

