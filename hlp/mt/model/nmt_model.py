from model import transformer as _transformer
import tensorflow as tf
from config import get_config as _config


def get_model(vocab_size_source, vocab_size_target):
    """获取模型"""
    transformer = _transformer.Transformer(_config.num_layers, _config.d_model, _config.num_heads, _config.dff,
                                           vocab_size_source + 1, vocab_size_target + 1,
                                           pe_input=vocab_size_source + 1,
                                           pe_target=vocab_size_target + 1,
                                           rate=_config.dropout_rate)
    return transformer


