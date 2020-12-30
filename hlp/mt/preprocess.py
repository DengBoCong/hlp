import re

from hlp.mt.common.text_vectorize import get_encoded_sequences_path, get_tokenizer_path, get_tokenizer_mode
from hlp.mt.config import get_config as _config
from hlp.mt.common import text_vectorize
from hlp.mt.common.text_split import preprocess_sentences
from hlp.mt.common.load_dataset import load_sentences


def _count_words(sentences):
    """输入句子列表，使用空格分隔返回单词数"""
    count = 0
    for s in sentences:
        s = re.split(r' +', s)
        count += len(s)
    return count


def train_preprocess():
    # 获取source、target编码模式，字典保存路径，编码句子保存路径
    source_mode = get_tokenizer_mode(_config.source_lang)
    target_mode = get_tokenizer_mode(_config.target_lang)

    source_tokenizer_path = get_tokenizer_path(_config.source_lang, source_mode)
    target_tokenizer_path = get_tokenizer_path(_config.target_lang, target_mode)

    source_sequences_path_train = get_encoded_sequences_path(_config.source_lang, postfix='_train')
    target_sequences_path_train = get_encoded_sequences_path(_config.target_lang, postfix='_train')
    source_sequences_path_val = get_encoded_sequences_path(_config.source_lang, postfix='_val')
    target_sequences_path_val = get_encoded_sequences_path(_config.target_lang, postfix='_val')

    # 加载句子
    print('加载训练数据集...')
    source_sentences, target_sentences = load_sentences(_config.path_to_train_file, _config.num_sentences)

    # 加载验证集
    if _config.validation_data == "True":
        print('加载验证数据集...')
        source_sentences_val, target_sentences_val = load_sentences(_config.path_to_val_file,
                                                                    _config.num_validate_sentences)

    print('已加载句子数量:%d' % _config.num_sentences)
    # 计算语料词数
    num_words = _count_words(source_sentences)
    print('源语料(%s)词数：%d' % (_config.source_lang, num_words))

    # 预处理句子
    print('预处理训练数据集...')
    source_sentences = preprocess_sentences(source_sentences, _config.source_lang, source_mode)
    target_sentences = preprocess_sentences(target_sentences, _config.target_lang, target_mode)

    if _config.validation_data == "True":
        print('预处理验证数据集...')
        source_sentences_val = preprocess_sentences(source_sentences_val, _config.source_lang, source_mode)
        target_sentences_val = preprocess_sentences(target_sentences_val, _config.target_lang, target_mode)

    # 生成及保存字典
    print('正在生成、保存源语言(%s)字典(分词方式:%s)...' % (_config.source_lang, _config.en_tokenize_type))
    tokenizer_source, vocab_size_source = text_vectorize.create_and_save_tokenizer(source_sentences,
                                                                                   source_tokenizer_path,
                                                                                   _config.source_lang,
                                                                                   source_mode)
    print('源语言字典大小:%d' % vocab_size_source)

    print('正在生成、保存目标语言(%s)字典(分词方式:%s)...' % (_config.target_lang, _config.zh_tokenize_type))
    tokenizer_target, vocab_size_target = text_vectorize.create_and_save_tokenizer(target_sentences,
                                                                                   target_tokenizer_path,
                                                                                   _config.target_lang,
                                                                                   target_mode)
    print('目标语言字典大小:%d' % vocab_size_target)

    # 编码句子
    print("正在编码训练集句子...")
    max_sequence_length_source = text_vectorize.encode_and_save(sentences=source_sentences, tokenizer=tokenizer_source,
                                                                save_path=source_sequences_path_train,
                                                                language=_config.source_lang, mode=source_mode)
    max_sequence_length_target = text_vectorize.encode_and_save(sentences=target_sentences, tokenizer=tokenizer_target,
                                                                save_path=target_sequences_path_train,
                                                                language=_config.target_lang, mode=target_mode)
    print('最大源语言(%s)句子长度:%d' % (_config.source_lang, max_sequence_length_source))
    print('最大目标语言(%s)句子长度:%d' % (_config.target_lang, max_sequence_length_target))

    if _config.validation_data == "True":
        print("正在编码验证集句子...")
        _ = text_vectorize.encode_and_save(sentences=source_sentences_val, tokenizer=tokenizer_source,
                                           save_path=source_sequences_path_val, language=_config.source_lang,
                                           mode=source_mode)
        _ = text_vectorize.encode_and_save(sentences=target_sentences_val, tokenizer=tokenizer_target,
                                           save_path=target_sequences_path_val, language=_config.target_lang,
                                           mode=target_mode)
    print("语料处理完成.\n")

    return vocab_size_source, vocab_size_target


if __name__ == '__main__':
    train_preprocess()
