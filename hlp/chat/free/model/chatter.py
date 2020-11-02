import os
import sys
import time
import tensorflow as tf
from pathlib import Path
import common.data_utils as _data
import config.get_config as _config
from utils.beamsearch import BeamSearch


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

    def train(self, checkpoint, dict_fn, data_fn, max_train_data_size):
        """
        对模型进行训练
        """
        print('训练开始，正在准备数据中...')
        input_tensor, target_tensor, _, dataset, steps_per_epoch, checkpoint_prefix = \
            _data.load_data(dict_fn=dict_fn,
                            data_fn=data_fn,
                            start_sign=_config.start_sign,
                            end_sign=_config.end_sign,
                            checkpoint_dir=self.checkpoint_dir,
                            max_train_data_size=max_train_data_size)

        for epoch in range(_config.epochs):
            print('Epoch {}/{}'.format(epoch + 1, _config.epochs))
            start_time = time.time()
            self._init_loss_accuracy()

            step_loss = [0]
            batch_sum = 0
            sample_sum = 0

            for (batch, (inp, tar, weight)) in enumerate(dataset.take(steps_per_epoch)):
                self._train_step(inp, tar, weight, step_loss)
                batch_sum = batch_sum + len(inp)
                sample_sum = steps_per_epoch * len(inp)
                print('\r', '{}/{} [==================================]'.format(batch_sum, sample_sum), end='',
                      flush=True)

            step_time = (time.time() - start_time)
            sys.stdout.write(' - {:.4f}s/step - loss: {:.4f}\n'
                             .format(step_time, step_loss[0]))
            sys.stdout.flush()
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('训练结束')

    def respond(self, req):
        # 对req进行初步处理
        inputs, dec_input = _data.preprocess_request(sentence=req, token=self.token)

        self.beam_search_container.init_all_inner_variables(inputs=inputs, dec_input=dec_input)
        inputs, dec_input = self.beam_search_container.expand_beam_size_inputs()
        for t in range(_config.max_length_tar):
            predictions = self._create_predictions(inputs, dec_input, t)
            self.beam_search_container.add(predictions=predictions, end_sign=self.token.get('end'))
            if self.beam_search_container.beam_size == 0:
                break

            inputs, dec_input = self.beam_search_container.expand_beam_size_inputs()
        beam_search_result = self.beam_search_container.get_result(top_k=3)
        result = ''
        # 从容器中抽取序列，生成最终结果
        for i in range(len(beam_search_result)):
            temp = beam_search_result[i].numpy()
            text = _data.sequences_to_texts(temp, self.token)
            text[0] = text[0].replace(_config.start_sign, '').replace(_config.end_sign, '').replace(' ', '')
            result = '<' + text[0] + '>' + result
        return result
