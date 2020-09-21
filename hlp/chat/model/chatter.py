import os
import sys
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

    def __init__(self, checkpoint_dir):
        """
        Transformer聊天器初始化，用于加载模型
        """
        self.checkpoint_dir = checkpoint_dir
        self.input_tensor, self.input_token, self.target_tensor, self.target_token = _data.load_dataset()
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

    def init_loss_accuracy(self):
        """
        初始化损失
        """
        pass

    def train_step(self, inp, tar, step_loss):
        """
        模型训练步方法，需要返回时间步损失
        """
        pass

    def create_predictions(self, inputs, dec_input, t):
        """
        使用模型预测下一个Token的id
        """
        pass

    def train(self, checkpoint):
        """
        对模型进行训练
        """
        dataset, checkpoint_prefix, steps_per_epoch = self.treat_dataset()

        for epoch in range(_config.epochs):
            print('当前训练epoch为：{}'.format(epoch + 1))
            start_time = time.time()

            self.init_loss_accuracy()

            step_loss = [0]
            for (batch, (inp, tar)) in enumerate(dataset.take(steps_per_epoch)):
                self.train_step(inp, tar, step_loss)

            step_time = (time.time() - start_time)
            print('当前epoch耗时：{:.4f}s：'.format(step_time))
            print('当前epoch损失：{:.4f}'.format(step_loss[0]))
            checkpoint.save(file_prefix=checkpoint_prefix)
            sys.stdout.flush()

        print('训练结束')

    def respond(self, req):
        # 对req进行初步处理
        inputs, dec_input = self.pre_treat_inputs(req)
        result = ''
        for t in range(_config.max_length_tar):
            predicted_id, dec_input = self.create_predictions(inputs, dec_input, t)
            if self.target_token.index_word.get(predicted_id.numpy()) == 'end':
                break
            result += self.target_token.index_word.get(predicted_id.numpy(), '')

        return result

    def stop(self):
        """ 结束聊天

        可以做一些清理工作
        :return:
        """
        pass

    def pre_treat_inputs(self, sentence):
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

    def treat_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.input_tensor, self.target_tensor)).cache().shuffle(
            _config.BUFFER_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(_config.BATCH_SIZE, drop_remainder=True)
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        print('训练开始，正在准备数据中...')
        step_per_epoch = len(self.input_tensor) // _config.BATCH_SIZE

        return dataset, checkpoint_prefix, step_per_epoch

    # def treat_predictions(self, predictions):
    #     # 取概率最大的值
    #     predicted_id = tf.cast(tf.argmax(predictions), tf.int32).numpy()
    #     # 返回对应token和最大的值
    #     return self.target_token.word_index.get(predicted_id, 'end'), predicted_id

class BeamContainer(object):
    def __init__(self, beam_size, max_length, worst_score, length_penalty):
        """
        初始化BeamSearch的序列容器
        """
        self.beam_size = beam_size
        self.max_length = max_length - 1
        self.remain = []
        self.worst_score = worst_score
        self.length_penalty = length_penalty

    def __len__(self):
        """
        已存在BeamSearch的序列容器的大小
        """
        return len(self.remain)

    def add(self, hyp, sum_logprobs):
        """
        往容器中添加序列，并行添加
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.beam_size or score > self.worst_score:
            self.remain.append((score, hyp))
            if len(self) > self.beam_size:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.remain)])
                del self.remain[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        相关样本是否已经完成生成
        best_sum_logprobs是新的候选序列中最高得分
        """
        if len(self) < self.beam_size:
            return False
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            # 是否最高分比当前保存的最低分还差
            ret = self.worst_score >= cur_score
            return ret
