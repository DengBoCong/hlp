import tensorflow as tf

import hlp.mt.common.text_vectorize
from hlp.mt.model import transformer as _transformer
from hlp.mt.config import get_config as _config
from hlp.mt.common import text_vectorize
from hlp.utils import optimizers as _optimizers


def create_model(vocab_size_source, vocab_size_target):
    """获取模型"""
    transformer = _transformer.Transformer(_config.num_layers,
                                           _config.d_model,
                                           _config.num_heads,
                                           _config.dff,
                                           vocab_size_source + 1,
                                           vocab_size_target + 1,
                                           pe_input=vocab_size_source + 1,
                                           pe_target=vocab_size_target + 1,
                                           rate=_config.dropout_rate)
    return transformer


def load_model():
    """
    进行翻译或评估前数据恢复工作
    """
    # 获取字典保存路径
    source_mode = hlp.mt.common.text_vectorize.get_tokenizer_mode(_config.source_lang)
    target_mode = hlp.mt.common.text_vectorize.get_tokenizer_mode(_config.target_lang)
    source_tokenizer_path = hlp.mt.common.text_vectorize.get_tokenizer_path(_config.source_lang, source_mode)
    target_tokenizer_path = hlp.mt.common.text_vectorize.get_tokenizer_path(_config.target_lang, target_mode)
    # 加载源语言字典
    print("正在加载源语言(%s)字典..." % _config.source_lang)
    tokenizer_source, vocab_size_source = text_vectorize.load_tokenizer(source_tokenizer_path,
                                                                        _config.source_lang, source_mode)
    print('源语言字典大小:%d' % vocab_size_source)
    print('源语言字典加载完毕！\n')

    # 加载目标语言字典
    print("正在加载目标语言(%s)字典..." % _config.target_lang)
    tokenizer_target, vocab_size_target = text_vectorize.load_tokenizer(target_tokenizer_path,
                                                                        _config.target_lang, target_mode)
    print('目标语言字典大小:%d' % vocab_size_target)
    print('目标语言字典加载完毕！\n')

    # 创建模型及相关变量
    learning_rate = _optimizers.CustomSchedule(_config.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    transformer = create_model(vocab_size_source, vocab_size_target)

    return transformer, optimizer, tokenizer_source, tokenizer_target
