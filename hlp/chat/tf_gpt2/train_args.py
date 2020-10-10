import tensorflow as tf
import random
import numpy as np
import argparse

def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--vocab_path', default='vocab/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--train_raw_path', default='data/data.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--raw', action='store_false', help='是否对原始训练语料做tokenize。若尚未对原始训练语料进行tokenize，则指定该参数')
    parser.add_argument('--epochs', default=1, type=int, required=False, help='训练的轮次')
    parser.add_argument('--batch_size', default=6, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--dialogue_model_output_path', default='dialogue_model/', type=str, required=False,
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

pad_id = 0
def collate_fn(batch):#对齐input
    """
    计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
    :param batch:
    :return:
    """
    global pad_id
    input_ids = []
    btc_size = len(batch)
    max_input_len = 0  # 该batch中最长的input，用于该batch的数据对齐
    # 计算该batch中input的最大长度
    for btc_idx in range(btc_size):
        if max_input_len < len(batch[btc_idx]):
            max_input_len = len(batch[btc_idx])
    # 使用pad_id对小于max_input_len的input_id进行补全
    for btc_idx in range(btc_size):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend(['0'] * (max_input_len - input_len))
    # print(input_ids)
    # print(input_ids.shape)
    return input_ids
