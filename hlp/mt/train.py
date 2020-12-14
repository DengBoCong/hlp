
from hlp.mt import trainer
from hlp.mt import preprocess as _pre
from hlp.mt.model import nmt_model
from hlp.mt.config import get_config as _config


def main():
    # 进行训练所需的数据处理
    vocab_size_source, vocab_size_target = _pre.train_preprocess()

    # 创建模型及相关变量
    model = nmt_model.get_model(vocab_size_source, vocab_size_target)

    # 开始训练
    trainer.train(model,
                  validation_data=_config.validation_data,
                  validation_split=1-_config.train_size,
                  validation_freq=_config.validation_freq)


if __name__ == '__main__':
    main()

