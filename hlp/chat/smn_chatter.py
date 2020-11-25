import os
import sys
import time
import pysolr
import model.smn as smn
import tensorflow as tf
sys.path.append(sys.path[0][:-10])
from common.utils import CmdParser
from common.utils import log_operator
import common.data_utils as data_utils
import config.get_config as get_config


class SMNChatter():
    """
    SMN的聊天器
    """

    def __init__(self, units: int, vocab_size: int, execute_type: str, dict_fn: str,
                 embedding_dim: int, checkpoint_dir: int, max_utterance: int, max_sentence: int,
                 learning_rate: float, database_fn: str, solr_server: str):
        """
        SMN聊天器初始化，用于加载模型
        Args:
            units: 单元数
            vocab_size: 词汇量大小
            execute_type: 对话执行模式
            dict_fn: 保存字典路径
            embedding_dim: 嵌入层维度
            checkpoint_dir: 检查点保存目录路径
            max_utterance: 每轮句子数量
            max_sentence: 单个句子最大长度
            learning_rate: 学习率
            database_fn: 候选数据库路径
        Returns:
        """
        self.dict_fn = dict_fn
        self.checkpoint_dir = checkpoint_dir
        self.max_utterance = max_utterance
        self.max_sentence = max_sentence
        self.database_fn = database_fn
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.solr = pysolr.Solr(url=solr_server, always_commit=True, timeout=10)
        self.train_loss = tf.keras.metrics.Mean()

        self.model = smn.smn(units=units, vocab_size=vocab_size,
                             embedding_dim=embedding_dim,
                             max_utterance=self.max_utterance,
                             max_sentence=self.max_sentence)

        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, )

        ckpt = os.path.exists(checkpoint_dir)
        if not ckpt:
            os.makedirs(checkpoint_dir)

        if execute_type == "chat":
            print('正在从“{}”处加载字典...'.format(self.dict_fn))
            self.token = data_utils.load_token_dict(dict_fn=self.dict_fn)
        print('正在检查是否存在检查点...')
        if ckpt:
            print('存在检查点，正在从“{}”中加载检查点...'.format(checkpoint_dir))
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        else:
            if execute_type == "train":
                print('不存在检查点，正在train模式...')
            else:
                print('不存在检查点，请先执行train模式，再进入chat模式')
                exit(0)

        logger = log_operator(level=10)
        logger.info("启动SMN聊天器，执行类别为：{}，模型参数配置为：embedding_dim：{}，"
                    "max_sentence：{}，max_utterance：{}，units：{}，vocab_size：{}，"
                    "learning_rate：{}".format(execute_type, embedding_dim, max_sentence,
                                              max_utterance, units, vocab_size, learning_rate))

    def train(self, epochs: int, data_fn: str, max_train_data_size: int = 0, max_valid_data_size: int = 0):
        """
        训练功能
        Args:
            epochs: 训练执行轮数
            data_fn: 数据文本路径
            max_train_data_size: 最大训练数据量
            max_valid_data_size: 最大验证数据量
        Returns:
        """
        # 处理并加载训练数据，
        dataset, tokenizer, checkpoint_prefix, steps_per_epoch = \
            data_utils.smn_load_train_data(dict_fn=self.dict_fn,
                                           data_fn=data_fn,
                                           checkpoint_dir=self.checkpoint_dir,
                                           max_utterance=self.max_utterance,
                                           max_sentence=self.max_sentence,
                                           max_train_data_size=max_train_data_size)

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            start_time = time.time()
            self.train_loss.reset_states()

            sample_sum = 0
            batch_sum = 0

            for (batch, (utterances, response, label)) in enumerate(dataset.take(steps_per_epoch)):
                with tf.GradientTape() as tape:
                    scores = self.model(inputs=[utterances, response])
                    loss = tf.keras.losses. \
                        SparseCategoricalCrossentropy(from_logits=True,
                                                      reduction=tf.keras.losses.Reduction.AUTO)(label, scores)
                gradient = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
                self.train_loss(loss)

                sample_num = len(utterances)
                batch_sum += sample_num
                sample_sum = steps_per_epoch * sample_num
                print('\r', '{}/{} [==================================]'.format(batch_sum, sample_sum),
                      end='', flush=True)

            r2_1, _ = self.evaluate(valid_fn=data_fn,
                                    tokenizer=tokenizer,
                                    max_valid_data_size=max_valid_data_size)

            step_time = time.time() - start_time
            sys.stdout.write(' - {:.4f}s/step - loss: {:.4f} - R2@1：{:0.3f}\n'
                             .format(step_time, self.train_loss.result(), r2_1))
            sys.stdout.flush()
            self.checkpoint.save(file_prefix=checkpoint_prefix)

    def evaluate(self, valid_fn: str, dict_fn: str = "", tokenizer: tf.keras.preprocessing.text.Tokenizer = None,
                 max_turn_utterances_num: int = 10, max_valid_data_size: int = 0):
        """
        验证功能，注意了dict_fn和tokenizer两个比传其中一个
        Args:
            valid_fn: 验证数据集路径
            dict_fn: 字典路径
            tokenizer: 分词器
            max_turn_utterances_num: 最大训练数据量
            max_valid_data_size: 最大验证数据量
        Returns:
            r2_1, r10_1
        """
        token_dict = None
        step = max_valid_data_size // max_turn_utterances_num
        if max_valid_data_size == 0:
            return None
        if dict_fn is not "":
            token_dict = data_utils.load_token_dict(dict_fn)
        # 处理并加载评价数据，注意，如果max_valid_data_size传
        # 入0，就直接跳过加载评价数据，也就是说只训练不评价
        valid_dataset = data_utils.load_smn_valid_data(data_fn=valid_fn,
                                                       max_sentence=self.max_sentence,
                                                       max_utterance=self.max_utterance,
                                                       token_dict=token_dict,
                                                       tokenizer=tokenizer,
                                                       max_turn_utterances_num=max_turn_utterances_num,
                                                       max_valid_data_size=max_valid_data_size)

        scores = tf.constant([], dtype=tf.float32)
        labels = tf.constant([], dtype=tf.int32)
        for (batch, (utterances, response, label)) in enumerate(valid_dataset.take(step)):
            score = self.model(inputs=[utterances, response])
            score = tf.nn.softmax(score, axis=-1)
            labels = tf.concat([labels, label], axis=0)
            scores = tf.concat([scores, score[:, 1]], axis=0)

        r10_1 = self._metrics_rn_1(scores, labels, num=10)
        r2_1 = self._metrics_rn_1(scores, labels, num=2)
        return r2_1, r10_1

    def respond(self, req: str):
        """
        对外部聊天请求进行回复
        子类需要利用模型进行推断和搜索以产生回复。
        Args:
            req: 输入的语句
        Returns: 系统回复字符串
        """
        self.solr.ping()
        history = req[-self.max_utterance:]
        pad_sequences = [0] * self.max_sentence
        utterance = data_utils.dict_texts_to_sequences(history, self.token)
        utterance_len = len(utterance)

        # 如果当前轮次中的历史语句不足max_utterances数量，需要在尾部进行填充
        if utterance_len != self.max_utterance:
            utterance += [pad_sequences] * (self.max_utterance - utterance_len)
        utterance = tf.keras.preprocessing.sequence.pad_sequences(utterance, maxlen=self.max_sentence,
                                                                  padding="post").tolist()

        tf_idf = data_utils.get_tf_idf_top_k(history)
        query = "{!func}sum("
        for key in tf_idf:
            query += "product(idf(utterance," + key + "),tf(utterance," + key + ")),"
        query += ")"
        candidates = self.solr.search(q=query, start=0, rows=10).docs
        candidates = [candidate['utterance'][0] for candidate in candidates]

        if candidates is None:
            return "Sorry! I didn't hear clearly, can you say it again?"
        else:
            utterances = [utterance] * len(candidates)
            responses = data_utils.dict_texts_to_sequences(candidates, self.token)
            responses = tf.keras.preprocessing.sequence.pad_sequences(responses, maxlen=self.max_sentence,
                                                                      padding="post")
            utterances = tf.convert_to_tensor(utterances)
            responses = tf.convert_to_tensor(responses)
            scores = self.model(inputs=[utterances, responses])
            index = tf.argmax(scores[:, 0])

            return candidates[index]

    def _metrics_rn_1(self, scores: float, labels: tf.Tensor, num: int = 10):
        """
        计算Rn@k指标
        Args:
            scores: 训练所得分数
            labels: 数据标签
            num: n
        Returns:
        """
        total = 0
        correct = 0
        for i in range(len(labels)):
            if labels[i] == 1:
                total = total + 1
                sublist = scores[i:i + num]
                if max(sublist) == scores[i]:
                    correct = correct + 1
        return float(correct) / total


def get_chatter(execute_type):
    """
    初始化要使用的聊天器
    Args:
        execute_type: 对话执行模型
    Returns:
        chatter: 返回对应的聊天器
    """
    chatter = SMNChatter(units=get_config.smn_units, vocab_size=get_config.smn_vocab_size,
                         execute_type=execute_type, dict_fn=get_config.smn_dict_fn, solr_server=get_config.solr_server,
                         embedding_dim=get_config.smn_embedding_dim, checkpoint_dir=get_config.smn_checkpoint,
                         max_utterance=get_config.smn_max_utterance, max_sentence=get_config.smn_max_sentence,
                         learning_rate=get_config.smn_learning_rate, database_fn=get_config.candidate_database)

    return chatter


def main():
    parser = CmdParser(version='%smn chatbot V1.0')
    parser.add_option("-t", "--type", action="store", type="string",
                      dest="type", default="pre_treat",
                      help="execute type, train/chat")
    (options, args) = parser.parse_args()

    if options.type == 'train':
        chatter = get_chatter(execute_type=options.type)
        chatter.train(epochs=get_config.epochs,
                      data_fn=get_config.ubuntu_tokenized_data,
                      max_train_data_size=get_config.smn_max_train_data_size,
                      max_valid_data_size=get_config.smn_max_valid_data_size)

    elif options.type == 'pre_treat':
        data_utils.creat_index_dataset(data_fn=get_config.ubuntu_tokenized_data,
                                       solr_sever=get_config.solr_server,
                                       max_database_size=get_config.smn_max_database_size)

    elif options.type == 'evaluate':
        chatter = get_chatter(execute_type=options.type)
        r2_1, r10_1 = chatter.evaluate(valid_fn=get_config.ubuntu_valid_data, dict_fn=get_config.smn_dict_fn,
                                       max_valid_data_size=get_config.smn_max_valid_data_size)
        print("指标：R2@1-{:0.3f}，R10@1-{:0.3f}".format(r2_1, r10_1))

    elif options.type == 'chat':
        chatter = get_chatter(execute_type=options.type)
        history = []  # 用于存放历史对话
        print("Agent: 你好！结束聊天请输入ESC。")
        while True:
            req = input("User: ")
            if req == "ESC":
                print("Agent: 再见！")
                exit(0)
            history.append(req)
            response = chatter.respond(req=history)
            print("Agent: ", response)
    else:
        parser.error(msg='')


if __name__ == '__main__':
    """
    SMN入口：指令需要附带运行参数
    cmd：python smn_chatter.py -t/--type [执行模式]
    执行类别：pre_treat/train/evaluate/chat

    chat模式下运行时，输入ESC即退出对话
    """
    main()
