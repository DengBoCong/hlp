from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
import os
import numpy as np
import argparse
from datetime import datetime
import logging
from transformers import BertTokenizer
import train_args
import train_tf
import predict_arg
PAD = '[PAD]'
pad_id = 0

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):

    assert len(logits.shape) == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, len(logits))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断

        indices_to_remove = logits < tf.raw_ops.TopKV2(input=logits, k=top_k,sorted=True,name=None)[0][..., -1, None]#true就是概率不属于前八 要过滤掉
        

        promax_value,promax_index=tf.raw_ops.TopKV2(input=logits, k=top_k,sorted=True,name=None)
        promax_value=promax_value.numpy()
        promax_index=promax_index.numpy()
        #print('promax_index={}'.format(promax_index))
        logits=promax_value

        # logits=np.array(logits)
        # logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷
    if top_p > 0.0:
        sorted_logits= tf.sort(logits,direction = 'DESCENDING')  # 对logits进行递减排序--返回降序值
        sorted_indices=tf.argsort(logits,direction = 'DESCENDING') #返回对应降序的序列值
        print('sorted_logits={}'.format(sorted_logits))
        cumulative_probs=tf.reduce_sum(tf.nn.softmax(sorted_logits,-1),-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p   ##过滤作用
        # Shift the indices to the right to keep also the first token above the threshold
        print('sorted_indices_to_remove.shape={}'.format(sorted_indices_to_remove.shape))
        print(sorted_indices_to_remove)
        print(sorted_indices_to_remove[..., :-1])
        sorted_indices_to_remove=np.array(sorted_indices_to_remove)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits,promax_index


def main():
    args = predict_arg.set_interact_args()
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
    tokenizer = BertTokenizer(vocab_file=args.voca_path)
    vocab_size = len(tokenizer)

    args_train = train_args.setup_train_args()

    model, _ = train_tf.create_model(args_train, vocab_size)

    model.load_weights('model_weight')
    print("Model Restored..........................")
    # checkpoint = tf.train.Checkpoint(optimizer=optimizer,
    #                                  model=model)
    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print('加载完权重')
        # 存储聊天记录，每个utterance以token的id的形式进行存储
    history = []
    print('开始和chatbot聊天，输入CTRL + Z以退出')

    while True:
        try:
            text = input("user:")
            if args.save_samples_path:
                samples_file.write("user:{}\n".format(text))
            history.append(tokenizer.encode(text))   #把输入的文本变成 token id
            input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头

            for history_id, history_utr in enumerate(history[-args.max_history_len:]):##切片
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)  #加分割ID
                # print('input_ids={}'.format(input_ids))
            curr_input_tensor =tf.convert_to_tensor(input_ids,tf.int64)#完整的输入id
            #print('curr_input_tensor={}'.format(curr_input_tensor))
            generated = []
            # 最多生成max_len个token
            for _ in range(args.max_len):
                outputs = model(inputs=curr_input_tensor)
                next_token_logits = outputs[0][-1, :]
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率--减少重复
                # for id in set(generated):
                #     print('id = {}'.format(id))
                #     next_token_logits[id] = next_token_logits[id]/args.repetition_penalty
                next_token_logits = next_token_logits / args.temperature
                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits=np.array(next_token_logits)
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')##？？？
                #print('next_token_logits[100]={}'.format(next_token_logits[100]))
                next_token_logits=tf.convert_to_tensor(next_token_logits)
                filtered_logits,promax_index = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                # multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标

                #print('filtered_logits={}'.format(filtered_logits))
                next_token_id = tf.raw_ops.Multinomial(logits=[tf.nn.softmax(filtered_logits, -1)], num_samples=1)
                next_token=promax_index[next_token_id]
                #print('next_token={}'.format(next_token))


                if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                    break
                #print('next_token={}'.format(next_token))
                generated.append(next_token[0])
                #print('generated=============={}'.format(generated))
                curr_input_tensor = tf.concat([curr_input_tensor, next_token], 0)
                # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
                # print("his_text:{}".format(his_text))
            history.append(generated)
            #print("history:{}".format(history))
            #print(tokenizer.word_index)
            text = tokenizer.convert_ids_to_tokens(generated)
            print("chatbot:" + "".join(text))
            if args.save_samples_path:
                samples_file.write("chatbot:{}\n".format("".join(text)))
        except KeyboardInterrupt:
            if args.save_samples_path:
                samples_file.close()
            break


if __name__ == '__main__':
    main()
