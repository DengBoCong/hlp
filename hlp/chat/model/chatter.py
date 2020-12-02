import os
import sys
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import common.data_utils as data_utils

from hlp.utils.beamsearch import BeamSearch


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

    def train(self, checkpoint: tf.train.Checkpoint, dict_fn: str, data_fn: str, batch_size: int,
              buffer_size: int, max_train_data_size: int, epochs: int, max_valid_data_size: int,
              save_dir: str, valid_data_split: float = 0.0, valid_data_fn: str = "", valid_freq: int = 1):
        """
        对模型进行训练，验证数据集优先级为：预设验证文本>训练划分文本>无验证
        Args:
            checkpoint: 模型的检查点
            dict_fn: 字典路径
            data_fn: 数据文本路径
            buffer_size: Dataset加载缓存大小
            batch_size: Dataset加载批大小
            max_train_data_size: 最大训练数据量
            epochs: 执行训练轮数
            save_dir: 历史指标显示图片保存位置
            max_valid_data_size: 最大验证数据量
            valid_data_split: 用于从训练数据中划分验证数据，默认0.1
            valid_data_fn: 验证数据文本路径
            valid_freq: 验证频率
        Returns:
        """
        print('训练开始，正在准备数据中...')
        train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch, checkpoint_prefix = \
            data_utils.load_data(dict_fn=dict_fn, data_fn=data_fn, start_sign=self.start_sign,
                                 buffer_size=buffer_size, batch_size=batch_size,
                                 end_sign=self.end_sign, checkpoint_dir=self.checkpoint_dir,
                                 max_length=self.max_length, valid_data_split=valid_data_split,
                                 valid_data_fn=valid_data_fn, max_train_data_size=max_train_data_size,
                                 max_valid_data_size=max_valid_data_size)

        valid_epochs_count = 0  # 用于记录验证轮次
        history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}

        for epoch in range(epochs):
            valid_epochs_count += 1
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            start_time = time.time()
            self._init_loss_accuracy()

            step_loss = 0
            step_accuracy = 0
            batch_sum = 0
            sample_sum = 0

            for (batch, (inp, tar, weight)) in enumerate(train_dataset.take(steps_per_epoch)):
                step_loss, step_accuracy = self._train_step(inp, tar, weight)
                batch_sum = batch_sum + len(inp)
                sample_sum = steps_per_epoch * len(inp)
                print('\r', '{}/{} [==================================]'.format(batch_sum, sample_sum), end='',
                      flush=True)

            step_time = (time.time() - start_time)
            history['accuracy'].append(step_accuracy.numpy())
            history['loss'].append(step_loss.numpy())

            sys.stdout.write(' - {:.4f}s/step - train_loss: {:.4f} - train_accuracy: {:.4f}\n'
                             .format(step_time, step_loss, step_accuracy))
            sys.stdout.flush()
            checkpoint.save(file_prefix=checkpoint_prefix)

            if valid_dataset is not None and valid_epochs_count % valid_freq == 0:
                valid_loss, valid_accuracy = self._valid_step(valid_dataset=valid_dataset,
                                                              steps_per_epoch=valid_steps_per_epoch)
                history['val_accuracy'].append(valid_accuracy.numpy())
                history['val_loss'].append(valid_loss.numpy())

        self._show_history(history=history, save_dir=save_dir, valid_freq=valid_freq)
        print('训练结束')
        return history

    def _show_history(self, history, save_dir, valid_freq):
        """
        用于显示历史指标趋势以及保存历史指标图表图
        Args:
            history: 历史指标
            save_dir: 历史指标显示图片保存位置
            valid_freq: 验证频率
        Returns:
        """
        train_x_axis = [i + 1 for i in range(len(history['loss']))]
        valid_x_axis = [(i + 1) * valid_freq for i in range(len(history['val_loss']))]

        figure, axis = plt.subplots(1, 1)
        tick_spacing = 1
        if len(history['loss']) > 20:
            tick_spacing = len(history['loss']) // 20
        plt.plot(train_x_axis, history['loss'], label='loss', marker='.')
        plt.plot(train_x_axis, history['accuracy'], label='accuracy', marker='.')
        plt.plot(valid_x_axis, history['val_loss'], label='val_loss', marker='.', linestyle='--')
        plt.plot(valid_x_axis, history['val_accuracy'], label='val_accuracy', marker='.', linestyle='--')
        plt.xticks(valid_x_axis)
        plt.xlabel('epoch')
        plt.legend()

        axis.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        save_path = save_dir + time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime(time.time()))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path)
        plt.show()

    def _valid_step(self, valid_dataset, steps_per_epoch):
        """
        对模型进行训练，验证数据集优先级为：预设验证文本>训练划分文本>无验证
        Args:
            valid_dataset: 验证Dataset
            steps_per_epoch: 验证数据总共的步数
        Returns:
        """
        print("验证轮次")
        start_time = time.time()
        self._init_loss_accuracy()
        step_loss = 0
        step_accuracy = 0
        batch_sum = 0
        sample_sum = 0

        for (batch, (inp, tar)) in enumerate(valid_dataset.take(steps_per_epoch)):
            step_loss, step_accuracy = self._train_step(inp, tar)
            batch_sum = batch_sum + len(inp)
            sample_sum = steps_per_epoch * len(inp)
            print('\r', '{}/{} [==================================]'.format(batch_sum, sample_sum), end='',
                  flush=True)

        step_time = (time.time() - start_time)
        sys.stdout.write(' - {:.4f}s/step - valid_loss: {:.4f} - valid_accuracy: {:.4f}\n'
                         .format(step_time, step_loss, step_accuracy))
        sys.stdout.flush()

        return step_loss, step_accuracy

    def respond(self, req: str):
        """
        对外部聊天请求进行回复
        子类需要利用模型进行推断和搜索以产生回复。
        Args:
            req: 输入的语句
        Returns: 系统回复字符串
        """
        # 对req进行初步处理
        inputs, dec_input = data_utils.preprocess_request(sentence=req, token=self.token, max_length=self.max_length,
                                                          start_sign=self.start_sign, end_sign=self.end_sign)
        self.beam_search_container.reset(inputs=inputs, dec_input=dec_input)
        inputs, dec_input = self.beam_search_container.get_search_inputs()

        for t in range(self.max_length):
            predictions = self._create_predictions(inputs, dec_input, t)
            self.beam_search_container.expand(predictions=predictions, end_sign=self.token.get(self.end_sign))
            # 注意了，如果BeamSearch容器里的beam_size为0了，说明已经找到了相应数量的结果，直接跳出循环
            if self.beam_search_container.beam_size == 0:
                break

            inputs, dec_input = self.beam_search_container.get_search_inputs()

        beam_search_result = self.beam_search_container.get_result(top_k=3)
        result = ''
        # 从容器中抽取序列，生成最终结果
        for i in range(len(beam_search_result)):
            temp = beam_search_result[i].numpy()
            text = data_utils.sequences_to_texts(temp, self.token)
            text[0] = text[0].replace(self.start_sign, '').replace(self.end_sign, '').replace(' ', '')
            result = '<' + text[0] + '>' + result
        return result
