"""
对指定路径文档进行加载处理
主要类及方法

可在配置文件中对中英文分词方法进行选择配置
- 英文分词父类 EnPreprocess
    - 对文档进行加载处理
    - encode_sentence(sentence) 对句子进行编码
    - decode_sentence(sequence) 对index列表进行解码
- 英文分词子类 EnPreprocessBpe
    - 使用BPE对文档进行编码
- 英文分词子类 EnPreprocessTokenize
    - 使用单词分词方法进行编码

- 中文分词父类 ChPreprocess
    - 对文档进行加载处理
    - encode_sentence(sentence) 对句子进行编码
    - decode_sentence(sequence) 对index列表进行解码
- 中文分词子类 EnPreprocessTokenize
    - 使用字分词方法进行编码

- 训练集与验证集划分方法 split_batch(input_tensor, target_tensor)

- 检测检查点方法 check_point()

"""

import tensorflow_datasets as tfds
import config.get_config as _config
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from pathlib import Path
import os


class EnPreprocess(object):
    '''英文分词父类'''
    def __init__(self, path, num_sentences, start_word, end_word):
        '''子类__init__需要对句子进行预处理编码及生成字典'''
        self.start_word = start_word
        self.end_word = end_word
        with open(path, encoding='UTF-8') as file:
            lines = file.read().strip().split('\n')

        self.raw_sentences = [l.split('\t')[0] for l in lines[:num_sentences]]

    def encode_sentence(self, sentence):
        '''使用字典对句子进行编码'''
        pass

    def decode_sequence(self, sequence):
        pass


class EnPreprocessBpe(EnPreprocess):
    """
    采用了subword对英文句子进行分词处理

    类属性：
    raw_sentences：加载的未处理句子列表
    sentences:经处理的且加上开始结束标志的句子列表（未编码）
    sequences:在sentences基础上编码后的句子列表
    tokenizer：字典
    vocab_size：字典大小
    max_sequence_length:编码句子最大长度

    类方法：
    encode_sentence:使用字典对句子进行编码
    decode_sequence：使用字典对句子进行解码
    """
    def __init__(self, path, num_sentences, start_word, end_word ,
                 target_vocab_size=2**13, load=False, vocab_filename=r'./data/vocab'):
        """
        Args:
            path: 加载的文本路径
            num_sentences: 取出的句子数
            start_word: 开始标志
            end_word: 结束标志
            target_vocab_size:目标字典大小
            load: 是否加载保存的字典
            vocab_filename: 保存字典的路径
        """
        super(EnPreprocessBpe, self).__init__(path, num_sentences, start_word, end_word)

        # 对句子进行预处理，加上开始结束标志
        self.sentences = [self.__preprocess_sentence(s) for s in self.raw_sentences]
        print('选择的英文分词配置为：BPE')
        # 得到编码的句子及字典
        if load:
            tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_filename)
        else:
            tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                self.sentences, target_vocab_size=target_vocab_size,
                reserved_tokens=[self.start_word, self.end_word])
            tokenizer.save_to_file(vocab_filename)
        self.sequences = [tokenizer.encode(s) for s in self.sentences]
        self.sequences = tf.keras.preprocessing.sequence.pad_sequences(self.sequences, padding='post')
        self.max_sequence_length = len(self.sequences[0])
        self.tokenizer = tokenizer

        # 词典大小
        self.vocab_size = self.tokenizer.vocab_size

    def __preprocess_sentence(self,sentence):
        sentence = self.start_word + ' ' + sentence + ' ' + self.end_word
        return sentence

    def encode_sentence(self, sentence):
        '''输入编码的句子，输出编码的句子'''
        sentence = self.__preprocess_sentence(sentence)
        return self.tokenizer.encode(sentence)

    def decode_sequence(self, sequence):
        '''输入编码的句子，输出解码的句子'''
        return self.tokenizer.decode(sequence)


class EnPreprocessTokenize(EnPreprocess):
    """
    采用了空格分割对英文句子进行分词处理
    未录入词显示为 'unk'

    类属性：
    raw_sentences：加载的未处理句子列表
    sentences:经处理的且加上开始结束标志的句子列表（未编码）
    sequences:在sentences基础上编码后的句子列表
    tokenizer：字典
    vocab_size：字典大小
    max_sequence_length:编码句子最大长度
    vocab:字典中所有的词的列表

    类方法：
    """
    def __init__(self, path, num_sentences, start_word, end_word ):
        """
        Args:
            path: 加载的文本路径
            num_sentences: 取出的句子数
            start_word: 开始标志
            end_word: 结束标志
        """
        super(EnPreprocessTokenize, self).__init__(path, num_sentences, start_word, end_word)
        print('选择的英文分词配置为：TOKENIZE（单词分词）')
        # 对句子进行预处理，加上开始结束标志
        self.sentences = [self.__preprocess_sentence(s) for s in self.raw_sentences]

        # 得到编码的句子及字典
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='unk')
        self.tokenizer.fit_on_texts(self.sentences)
        self.sequences = self.tokenizer.texts_to_sequences(self.sentences)
        self.sequences = tf.keras.preprocessing.sequence.pad_sequences(self.sequences, padding='post')
        self.max_sequence_length = len(self.sequences[0])

        # 词典大小
        self.vocab_size = len(self.tokenizer.word_index)

        # 词库
        self.vocab = [i for i in self.tokenizer.word_index]

    def __preprocess_sentence(self,sentence):
        '''对英语句子进行处理'''
        s = sentence.lower().strip()
        s = re.sub(r'([?.!,])', r' \1', s)  # 在?.!,前添加空格
        s = re.sub(r'[^a-zA-Z?,!.]+', " ", s)  # 将除字母及标点外的字符变为空格
        s = re.sub(r'[" "]+', " ", s)  # 合并连续的空格
        s = s.strip()

        s = self.start_word + ' ' + s + ' ' + self.end_word  # 给句子加上开始结束标志
        return s

    def encode_sentence(self, sentence):
        '''输入编码的句子，输出编码的句子'''
        sentence = self.__preprocess_sentence(sentence).split()
        sequence = []
        # 将句子中未出现在词库中的word替换为'unk'
        for w in sentence:
            if w in self.vocab:
                sequence.append(w)
            else:
                sequence.append('unk')
        sequence = [self.tokenizer.word_index[w] for w in sequence]
        return sequence

    def decode_sequence(self, sequence):
        '''输入编码的句子，输出解码的句子'''
        sentence = [self.tokenizer.index_word[idx] for idx in sequence]
        sentence = ' '.join(sentence)
        return sentence


class ChPreprocess(object):
    '''中文分词父类'''
    def __init__(self, path, num_sentences, start_word, end_word):
        '''子类__init__需要对句子进行预处理编码及生成字典'''
        self.start_word = start_word
        self.end_word = end_word
        with open(path, encoding='UTF-8') as file:
            lines = file.read().strip().split('\n')

        self.raw_sentences = [l.split('\t')[1] for l in lines[:num_sentences]]

    def encode_sentence(self, sentence):
        '''使用字典对句子进行编码'''
        pass

    def decode_sequence(self, sequence):
        pass


class ChPreprocessTokenize(ChPreprocess):
    '''使用单字分词法的中文分词子类'''
    def __init__(self, path, num_sentences, start_word, end_word):
        """
        Args:
            path: 加载的文本路径
            num_sentences: 取出的句子数
            start_word: 开始标志
            end_word: 结束标志
        """
        super(ChPreprocessTokenize, self).__init__(path, num_sentences, start_word, end_word)
        print('选择的中文分词配置为：TOKENIZE（字分词）')
        # 对句子进行预处理，加上开始结束标志
        self.sentences = [self.__preprocess_sentence(s) for s in self.raw_sentences]

        # 得到编码的句子及字典
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='unk')
        self.tokenizer.fit_on_texts(self.sentences)
        self.sequences = self.tokenizer.texts_to_sequences(self.sentences)
        self.sequences = tf.keras.preprocessing.sequence.pad_sequences(self.sequences, padding='post')
        self.max_sequence_length = len(self.sequences[0])

        # 词典大小
        self.vocab_size = len(self.tokenizer.word_index)

        # 词库
        self.vocab = [i for i in self.tokenizer.word_index]

    def __preprocess_sentence(self,sentence):
        '''对中文句子进行处理'''
        s = sentence.strip()
        s = ' '.join(s)
        s = s.strip()
        s = '<start> ' + s + ' <end>'  # 给句子加上开始结束标志
        return s

        s = self.start_word + ' ' + s + ' ' + self.end_word  # 给句子加上开始结束标志
        return s

    def encode_sentence(self, sentence):
        '''输入编码的句子，输出编码的句子'''
        sentence = self.__preprocess_sentence(sentence).split()
        sequence = []
        # 将句子中未出现在词库中的word替换为'unk'
        for w in sentence:
            if w in self.vocab:
                sequence.append(w)
            else:
                sequence.append('unk')
        sequence = [self.tokenizer.word_index[w] for w in sequence]
        return sequence

    def decode_sequence(self, sequence):
        '''输入编码的句子，输出解码的句子'''
        sentence = [self.tokenizer.index_word[idx.numpy()] for idx in sequence
                    if idx != [self.tokenizer.word_index[self.start_word]]]
        sentence = ''.join(sentence)
        return sentence


def split_batch(input_tensor, target_tensor):
    '''将输入输出句子进行训练集及验证集的划分,返回张量'''
    x_train, x_test, y_train, y_test = train_test_split(input_tensor, target_tensor, test_size=_config.test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    train_dataset = train_dataset.shuffle(_config.BUFFER_SIZE).batch(_config.BATCH_SIZE, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.shuffle(_config.BUFFER_SIZE).batch(_config.BATCH_SIZE, drop_remainder=True)
    return train_dataset, val_dataset


# temp_train_dataset, temp_val_dataset = split_batch(temp_input_sequences, temp_target_sequences)
# example_input_batch, example_target_batch = next(iter(temp_train_dataset))
# print(example_input_batch.shape, example_target_batch.shape)
# print('-'*20)

def check_point():
    '''检测检查点目录下是否有文件'''
    checkpoint_dir = _config.checkpoint_path
    is_exist = Path(checkpoint_dir)
    if not is_exist.exists():
        os.makedirs(checkpoint_dir, exist_ok=True)
    if_ckpt = tf.io.gfile.listdir(checkpoint_dir)
    return if_ckpt

