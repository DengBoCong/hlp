import os
import sys
import time
import tensorflow as tf
import common.data_utils as data_utils
from utils.beamsearch import BeamSearch


class Chatter(object):
    """"
    面向使用者的聊天器基类
    该类及其子类实现和用户间的聊天，即接收聊天请求，产生回复。
    不同模型或方法实现的聊天子类化该类。
    """

    def __init__(self, checkpoint_dir: str, beam_size: int, max_length: int):
        """
        聊天器初始化，用于加载模型
        Args:
            checkpoint_dir: 检查点保存目录路径
            beam_size: batch大小
            max_length: 单个句子最大长度
        Returns:
        """
        self.max_length = max_length
        self.checkpoint_dir = checkpoint_dir
        self.beam_search_container = BeamSearch(
            beam_size=beam_size,
            max_length=max_length,
            worst_score=0
        )

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        self.ckpt = tf.io.gfile.listdir(checkpoint_dir)

    def _init_loss_accuracy(self):
        """
        初始化损失
        """
        pass

    def _train_step(self, inp: tf.Tensor, tar: tf.Tensor, weight: int, step_loss: float):
        """
        模型训练步方法，需要返回时间步损失
        Args:
            inp: 输入序列
            tar: 目标序列
            weight: 样本权重序列
            step_loss: 每步损失
        Returns:
            step_loss: 每步损失
        """
        pass

    def _create_predictions(self, inputs: tf.Tensor, dec_input: tf.Tensor, t: int):
        """
        使用模型预测下一个Token的id
        Args:
            inputs: 对话中的问句
            dec_input: 对话中的答句
            t: 记录时间步
        Returns:
            predictions: 预测
        """
        pass

    def train(self, checkpoint: tf.train.Checkpoint, dict_fn: str,
              data_fn: str, max_train_data_size: int, epochs: int):
        """
        对模型进行训练
        Args:
            checkpoint: 模型的检查点
            dict_fn: 字典路径
            data_fn: 数据文本路径
            max_train_data_size: 最大训练数据量
            epochs: 执行训练轮数
        Returns:
        """
        print('训练开始，正在准备数据中...')
        input_tensor, target_tensor, _, dataset, steps_per_epoch, checkpoint_prefix = \
            data_utils.load_data(dict_fn=dict_fn,
                                 data_fn=data_fn,
                                 start_sign=self.start_sign,
                                 end_sign=self.end_sign,
                                 checkpoint_dir=self.checkpoint_dir,
                                 max_length=self.max_length,
                                 max_train_data_size=max_train_data_size)

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            start_time = time.time()
            self._init_loss_accuracy()

            step_loss = 0
            batch_sum = 0
            sample_sum = 0

            for (batch, (inp, tar, weight)) in enumerate(dataset.take(steps_per_epoch)):
                step_loss = self._train_step(inp, tar, weight, step_loss)
                batch_sum = batch_sum + len(inp)
                sample_sum = steps_per_epoch * len(inp)
                print('\r', '{}/{} [==================================]'.format(batch_sum, sample_sum), end='',
                      flush=True)

            step_time = (time.time() - start_time)
            sys.stdout.write(' - {:.4f}s/step - loss: {:.4f}\n'
                             .format(step_time, step_loss))
            sys.stdout.flush()
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('训练结束')

    def respond(self, req: str):
        """
        对外部聊天请求进行回复
        子类需要利用模型进行推断和搜索以产生回复。
        Args:
            req: 输入的语句
        Returns: 系统回复字符串
        """
        # 对req进行初步处理
        inputs, dec_input = data_utils.preprocess_request(sentence=req, token=self.token, max_length=self.max_length)
        self.beam_search_container.reset(inputs=inputs, dec_input=dec_input)
        inputs, dec_input = self.beam_search_container.expand_beam_size_inputs()

        for t in range(self.max_length):
            predictions = self._create_predictions(inputs, dec_input, t)
            self.beam_search_container.add(predictions=predictions, end_sign=self.token.get(self.end_sign))
            # 注意了，如果BeamSearch容器里的beam_size为0了，说明已经找到了相应数量的结果，直接跳出循环
            if self.beam_search_container.beam_size == 0:
                break

            inputs, dec_input = self.beam_search_container.expand_beam_size_inputs()

        beam_search_result = self.beam_search_container.get_result(top_k=3)
        result = ''
        # 从容器中抽取序列，生成最终结果
        for i in range(len(beam_search_result)):
            temp = beam_search_result[i].numpy()
            text = data_utils.sequences_to_texts(temp, self.token)
            text[0] = text[0].replace(self.start_sign, '').replace(self.end_sign, '').replace(' ', '')
            result = '<' + text[0] + '>' + result
        return result
