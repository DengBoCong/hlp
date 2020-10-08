import os
import sys
import copy
import time
import jieba
import tensorflow as tf
from pathlib import Path
import common.data_utils as _data
import config.get_config as _config


class Chatter(object):
    """"
    面向使用者的聊天器基类
    该类及其子类实现和用户间的聊天，即接收聊天请求，产生回复。
    不同模型或方法实现的聊天子类化该类。
    """

    def __init__(self, checkpoint_dir, beam_size):
        """
        Transformer聊天器初始化，用于加载模型
        """
        self.checkpoint_dir = checkpoint_dir
        self.input_tensor, self.input_token, self.target_tensor, self.target_token = _data.load_dataset()
        self.beam_search_container = BeamSearch(
            beam_size=beam_size,
            max_length=_config.max_length_tar,
            worst_score=0
        )
        is_exist = Path(checkpoint_dir)
        if not is_exist.exists():
            os.makedirs(checkpoint_dir, exist_ok=True)
        self.ckpt = tf.io.gfile.listdir(checkpoint_dir)

    def respond(self, req):
        """ 对外部聊天请求进行回复
        子类需要利用模型进行推断和搜索以产生回复。
        :param req: 外部聊天请求字符串
        :return: 系统回复字符串
        """
        pass

    def _init_loss_accuracy(self):
        """
        初始化损失
        """
        pass

    def _train_step(self, inp, tar, step_loss):
        """
        模型训练步方法，需要返回时间步损失
        """
        pass

    def _create_predictions(self, inputs, dec_input, t):
        """
        使用模型预测下一个Token的id
        """
        pass

    def train(self, checkpoint):
        """
        对模型进行训练
        """
        dataset, checkpoint_prefix, steps_per_epoch = self._treat_dataset()

        for epoch in range(_config.epochs):
            print('Epoch {}/{}'.format(epoch + 1, _config.epochs))
            start_time = time.time()

            self._init_loss_accuracy()

            step_loss = [0]
            batch_sum = 0
            sample_sum = 0
            for (batch, (inp, tar)) in enumerate(dataset.take(steps_per_epoch)):
                self._train_step(inp, tar, step_loss)
                batch_sum = batch_sum + len(inp)
                sample_sum = steps_per_epoch * len(inp)
                sys.stdout.write('{}/{} [==================================]'.format(batch_sum, sample_sum))
                sys.stdout.flush()

            step_time = (time.time() - start_time)
            sys.stdout.write(' - {:.4f}s/step - loss: {:.4f}\n'
                             .format(step_time, step_loss[0]))
            sys.stdout.flush()
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('训练结束')

    def respond(self, req):
        # 对req进行初步处理
        inputs, dec_input = self._pre_treat_inputs(req)
        self.beam_search_container.init_variables(inputs=inputs, dec_input=dec_input)
        inputs, dec_input = self.beam_search_container.get_variables()
        for t in range(_config.max_length_tar):
            predictions = self._create_predictions(inputs, dec_input, t)
            self.beam_search_container.add(predictions=predictions, end_sign=self.target_token.word_index.get('end'))
            if self.beam_search_container.beam_size == 0:
                break

            inputs, dec_input = self.beam_search_container.get_variables()
        beam_search_result = self.beam_search_container.get_result()
        result = ''
        # 从容器中抽取序列，生成最终结果
        for i in range(len(beam_search_result)):
            temp = beam_search_result[i].numpy()
            text = self.target_token.sequences_to_texts(temp)
            text[0] = text[0].replace('start', '').replace('end', '').replace(' ', '')
            result = '<' + text[0] + '>' + result
        return result

    def _pre_treat_inputs(self, sentence):
        # 分词
        sentence = " ".join(jieba.cut(sentence))
        # 添加首尾符号
        sentence = _data.preprocess_sentence(sentence)
        # 将句子转成token列表
        inputs = [self.input_token.word_index.get(i, 3) for i in sentence.split(' ')]
        # 填充
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=_config.max_length_inp, padding='post')
        # 转成Tensor
        inputs = tf.convert_to_tensor(inputs)
        # decoder的input就是开始符号
        dec_input = tf.expand_dims([self.target_token.word_index['start']], 0)
        return inputs, dec_input

    def _treat_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.input_tensor, self.target_tensor)).cache().shuffle(
            _config.BUFFER_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(_config.BATCH_SIZE, drop_remainder=True)
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        print('训练开始，正在准备数据中...')
        step_per_epoch = len(self.input_tensor) // _config.BATCH_SIZE

        return dataset, checkpoint_prefix, step_per_epoch


class BeamSearch(object):
    """
    BeamSearch使用说明：
    1.首先需要将问句编码成token向量并对齐，然后调用init_input方法进行初始化
    2.对模型要求能够进行批量输入
    3.BeamSearch使用实例已经集成到Chatter中，如果不进行自定义调用，
    可以将聊天器继承Chatter，在满足上述两点的基础之上设计_create_predictions方法，并调用BeamSearch
    """

    def __init__(self, beam_size, max_length, worst_score):
        """
        初始化BeamSearch的序列容器
        """
        self.remain_beam_size = beam_size
        self.max_length = max_length - 1
        self.remain_worst_score = worst_score
        self._reset_variables()

    def __len__(self):
        """
        已存在BeamSearch的序列容器的大小
        """
        return len(self.container)

    def init_variables(self, inputs, dec_input):
        """
        用来初始化输入
        :param inputs: 已经序列化的输入句子
        :param dec_input: 编码器输入序列
        :return: 无返回值
        """
        self.container.append((1, dec_input))
        self.inputs = inputs
        self.dec_inputs = dec_input

    def get_variables(self):
        """
        用来动态的更新模型的inputs和dec_inputs，以适配随着Beam Search
        结果的得出而变化的beam_size
        :return: requests, dec_inputs
        """
        # 生成多beam输入
        inputs = self.inputs
        for i in range(len(self) - 1):
            inputs = tf.concat([inputs, self.inputs], 0)
        requests = inputs
        # 生成多beam的decoder的输入
        temp = self.container[0][1]
        for i in range(1, len(self)):
            temp = tf.concat([temp, self.container[i][1]], axis=0)
        self.dec_inputs = copy.deepcopy(temp)
        return requests, self.dec_inputs

    def _reduce_end(self, end_sign):
        """
        当序列遇到了结束token，需要将该序列从容器中移除
        :return: 无返回值
        """
        for idx, (s, dec) in enumerate(self.container):
            temp = dec.numpy()
            if temp[0][-1] == end_sign:
                self.result.append(self.container[idx][1])
                del self.container[idx]
                self.beam_size -= 1

    def _reset_variables(self):
        """
        重置相关变量
        :return: 无返回值
        """
        self.beam_size = self.remain_beam_size
        self.worst_score = self.remain_worst_score
        self.container = []  # 保存中间状态序列的容器，元素格式为(score, sequence)类型为(float, [])
        self.result = []  # 用来保存已经遇到结束符的序列
        self.inputs = tf.constant(0, shape=(1, 1))
        self.dec_inputs = tf.constant(0, shape=(1, 1))  # 处理后的的编码器输入

    def add(self, predictions, end_sign):
        """
        往容器中添加预测结果，在本方法中对预测结果进行整理、排序的操作
        :param predictions: 传入每个时间步的模型预测值
        :return: 无返回值
        """
        remain = copy.deepcopy(self.container)
        for i in range(self.dec_inputs.shape[0]):
            for k in range(predictions.shape[-1]):
                if predictions[i][k] <= 0:
                    continue
                # 计算分数
                score = remain[i][0] * predictions[i][k]
                # 判断容器容量以及分数比较
                if len(self) < self.beam_size or score > self.worst_score:
                    self.container.append((score, tf.concat([remain[i][1], tf.constant([[k]], shape=(1, 1))], axis=-1)))
                    if len(self) > self.beam_size:
                        sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.container)])
                        del self.container[sorted_scores[0][1]]
                        self.worst_score = sorted_scores[1][0]
                    else:
                        self.worst_score = min(score, self.worst_score)
        self._reduce_end(end_sign=end_sign)

    def get_result(self):
        """
        获取最终beam个序列
        :return: beam个序列
        """
        result = self.result

        # 每轮回答之后，需要重置容器内部的相关变量值
        self._reset_variables()
        return result
