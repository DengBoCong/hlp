import tensorflow as tf
import random
import numpy as np
import argparse

pad_id = 0


def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', default='vocab/vocab_middle.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--chat_train_raw_path', default='chat_data/chat_data.txt', type=str, required=False,
                        help='原始训练语料')
    parser.add_argument('--train_raw_path', default='poem_data/poem_raw.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--chat_train_tokenized_path', default='chat_data/chat_train_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--train_tokenized_path', default='poem_data/poem_train_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--raw', action='store_false', help='是否对原始训练语料做tokenize。若尚未对原始训练语料进行tokenize，则指定该参数')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='训练batch size')
    parser.add_argument('--data_size', default=10014, type=int, required=False, help='训练总句子对长度')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--dialogue_model_output_path', default='poem_checkpoints/', type=str, required=False,
                        help='对话模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='预训练的GPT2模型的路径')
    parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=1, help="dataloader加载数据时使用的线程数量")
    # parser.add_argument('--max_len', type=int, default=60, help='每个utterance的最大长度,超过指定长度则进行截断')
    # parser.add_argument('--max_history_len', type=int, default=4, help="dialogue history的最大长度")
    return parser.parse_args()


def set_random_seed(args):
    """
    设置训练的随机种子
    """
    tf.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
