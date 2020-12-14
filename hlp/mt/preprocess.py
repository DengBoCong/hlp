import re

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
    """
    模型训练所需要的文本预处理
    - 加载句子
    - 预处理句子
    - 生成及保存字典
    - 编码句子

    """
    # 加载句子
    print('正在加载、预处理数据...')
    # en = _pre.load_single_sentences(_config.path_to_train_file_en, _config.num_sentences, column=1)
    # ch = _pre.load_single_sentences(_config.path_to_train_file_zh, _config.num_sentences, column=1)
    source_sentences, target_sentences = load_sentences(_config.path_to_train_file, _config.num_sentences)
    # 加载验证集
    if _config.validation_data == "True":
        source_sentences_val, target_sentences_val = load_sentences(_config.path_to_val_file,
                                                                    _config.num_validate_sentences)

    # 计算语料词数
    num_words = _count_words(source_sentences)
    print('源语料(%s)词数：%d' % (_config.source_lang, num_words))

    # 预处理句子
    source_sentences = preprocess_sentences(source_sentences, language=_config.source_lang)
    target_sentences = preprocess_sentences(target_sentences, language=_config.target_lang)
    if _config.validation_data == "True":
        source_sentences_val = preprocess_sentences(source_sentences_val, language=_config.source_lang)
        target_sentences_val = preprocess_sentences(target_sentences_val, language=_config.target_lang)
    print('已加载句子数量:%d' % _config.num_sentences)
    print('数据加载、预处理完毕！\n')

    # 生成及保存字典
    print('正在生成、保存源语言(%s)字典(分词方式:%s)...' % (_config.source_lang, _config.en_tokenize_type))
    tokenizer_source, vocab_size_source = text_vectorize.create_and_save_tokenizer(sentences=source_sentences,
                                                                                   language=_config.source_lang)
    print('生成英文字典大小:%d' % vocab_size_source)
    print('源语言字典生成、保存完毕！\n')

    print('正在生成、保存目标语言(%s)字典(分词方式:%s)...' % (_config.target_lang, _config.zh_tokenize_type))
    tokenizer_target, vocab_size_target = text_vectorize.create_and_save_tokenizer(sentences=target_sentences,
                                                                                   language=_config.target_lang)
    print('生成目标语言字典大小:%d' % vocab_size_target)
    print('目标语言字典生成、保存完毕！\n')

    # 编码句子
    print("正在编码训练集句子...")
    max_sequence_length_source = text_vectorize.encode_and_save(sentences=source_sentences,
                                                                tokenizer=tokenizer_source,
                                                                language=_config.source_lang)
    max_sequence_length_target = text_vectorize.encode_and_save(sentences=target_sentences,
                                                                tokenizer=tokenizer_target,
                                                                language=_config.target_lang)
    print('最大源语言(%s)句子长度:%d' % (_config.source_lang, max_sequence_length_source))
    print('最大目标语言(%s)句子长度:%d' % (_config.target_lang, max_sequence_length_target))
    if _config.validation_data == "True":
        print("正在编码验证集句子...")
        _ = text_vectorize.encode_and_save(sentences=source_sentences_val,
                                           tokenizer=tokenizer_source,
                                           language=_config.source_lang,
                                           postfix='_val')
        _ = text_vectorize.encode_and_save(sentences=target_sentences_val,
                                           tokenizer=tokenizer_target,
                                           language=_config.target_lang,
                                           postfix='_val')
    print("句子编码完毕！\n")

    return vocab_size_source, vocab_size_target


def main():
    train_preprocess()


if __name__ == '__main__':
    main()
