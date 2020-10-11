import tensorflow as tf
import numpy as np
import argparse

PAD = '[PAD]'
pad_id = 0

def set_interact_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='config/model_config_dialogue_small.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--log_path', default='data/interacting.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--voca_path', default='vocab/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=5, help="dialogue history的最大长度")
    return parser.parse_args()

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):

    assert len(logits.shape) == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, len(logits))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        #indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]#true就是概率不属于前八 要过滤掉
        indices_to_remove = logits < tf.raw_ops.TopKV2(input=logits, k=top_k,sorted=True,name=None)[0][..., -1, None]#true就是概率不属于前八 要过滤掉
        #print('排序')
        promax_value,promax_index=tf.raw_ops.TopKV2(input=logits, k=top_k,sorted=True,name=None)
        promax_value=promax_value.numpy()
        promax_index=promax_index.numpy()
        #print('promax_index={}'.format(promax_index))
        logits=promax_value

        # logits=np.array(logits)
        # logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    # 此功能暂未使用 先注释掉 后续增加
    # if top_p > 0.0:
    #     sorted_logits= tf.sort(logits,direction = 'DESCENDING')  # 对logits进行递减排序--返回降序值
    #     sorted_indices=tf.argsort(logits,direction = 'DESCENDING') #返回对应降序的序列值
    #     print('sorted_logits={}'.format(sorted_logits))
    #     #cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=1-)##累积概率
    #     cumulative_probs=tf.reduce_sum(tf.nn.softmax(sorted_logits,-1),-1)
    #     sorted_indices_to_remove = cumulative_probs > top_p   ##过滤作用
    #     print('sorted_indices_to_remove.shape={}'.format(sorted_indices_to_remove.shape))
    #     print(sorted_indices_to_remove)
    #     print(sorted_indices_to_remove[..., :-1])
    #     sorted_indices_to_remove=np.array(sorted_indices_to_remove)
    #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #     sorted_indices_to_remove[..., 0] = 0
    #
    #     indices_to_remove = sorted_indices[sorted_indices_to_remove]
    #     logits[indices_to_remove] = filter_value
    return logits,promax_index