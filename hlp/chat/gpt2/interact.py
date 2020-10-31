import tensorflow as tf
import os
import numpy as np
from datetime import datetime
from transformers import BertTokenizer
import train_args as train_args
import train as train
import interact_arg as interact_arg
from gpt2 import TFGPT2Model
from gpt2_config import GPT2Config
PAD = '[PAD]'
pad_id = 0


def main():
    args = interact_arg.set_interact_args()
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
    tokenizer = BertTokenizer(vocab_file=args.voca_path)
    config = GPT2Config()
    args_train = train_args.setup_train_args()
    model, n_ctx, optimizer = train.create_model(args_train, config)
    train.load_checkpoint(model, optimizer, args)
    # model.load_weights('./chat_checkpoints/model_weight').expect_partial()
    print("Model Restored..........................")

    # 存储聊天记录，每个utterance以token的id的形式进行存储
    history = []
    print('开始和chatbot聊天，输入CTRL + Z以退出')

    while True:
        try:
            text = input("user:")
            length = len(text)
            if args.save_samples_path:
                samples_file.write("user:{}\n".format(text))
            history.append(tokenizer.encode(text))  # 把输入的文本变成 token id
            input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头

            for history_id, history_utr in enumerate(history[-args.max_history_len:]):  ##切片
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)  # 加分割ID
            curr_input_tensor = tf.convert_to_tensor(input_ids, tf.int64)  # 完整的输入id
            generated = []
            # 最多生成max_len个token
            for _ in range(length):
                outputs = model(inputs=curr_input_tensor)
                next_token_logits = outputs[0][-1, :]  # （vocab,)
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率--减少重复
                # for id in set(generated):
                #     print('id = {}'.format(id))
                #     next_token_logits[id] = tf.divide(next_token_logits[id],args.repetition_penalty)
                # next_token_logits = tf.divide(next_token_logits , args.temperature)
                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                next_token_logits = np.array(next_token_logits)
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                # print('next_token_logits[100]={}'.format(next_token_logits[100]))
                next_token_logits = tf.convert_to_tensor(next_token_logits)
                filtered_logits, promax_index = interact_arg.top_k_top_p_filtering(next_token_logits, top_k=args.topk,
                                                                                   top_p=args.topp)
                # multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                logit_token = tf.convert_to_tensor(tf.nn.softmax(filtered_logits, -1)).numpy()
                next_token_id = tf.math.argmax(logit_token)
                next_token = promax_index[next_token_id]
                if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                    break
                generated.append(next_token)
                curr_input_tensor = tf.concat([curr_input_tensor, [next_token]], 0)
            history.append(generated)
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
