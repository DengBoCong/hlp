import tensorflow as tf
import transformers
import os
import random
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm #可以在控制台显示进度条
import logging
from sklearn.model_selection import train_test_split
from transformers import GPT2Config
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from transformers import BertTokenizer
from itertools import zip_longest, chain
#不能是一一对应 得是理解了之后的重新编写

from hlp.chat.GPT2 import preprocess_data
#数据处理
from hlp.chat.GPT2 import train_args

PAD='[PAD]'

def create_model(args, vocab_size):
    """

    :param args:
    :param vocab_size:字典大小
    :return:
    """
    print('配置模型参数')
    #model_config = GPT2Config.from_json_file('config/model_config_dialogue_small.json')

    print('创建model')
    model = TFGPT2LMHeadModel.from_pretrained('gpt2')

    #model = TFGPT2LMHeadModel.from_pretrained()#实例化一个类
    # 根据tokenizer的vocabulary调整GPT2模型的voca的大小
    return model, model.config.to_dict().get("n_ctx")

pad_id = 0
def calculate_loss_and_accuracy(outputs, labels):
    """
    计算非pad_id的平均loss和准确率
    :param outputs:
    :param labels:
    :param device:
    :return:
    """
    logits = outputs[0]  # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size]
    #print('logits.shape={}'.format(logits.shape))
    # 用前n-1个token，预测出第n个token
    # 用第i个token的prediction_score用来预测第i+1个token。
    # 假定有input有n个token，则shift_logits表示model中第[0,n-2]个token的prediction_score，shift_labels表示第[1，n-1]的label
    shift_logits = logits[:, :-1,]
    shift_labels = labels[:,1:]

    print('转换之后')
    shift_labels=tf.convert_to_tensor(shift_labels)
    shift_logits=tf.reshape(shift_logits,[-1, tf.convert_to_tensor(shift_logits).shape[-1]])
    shift_labels=tf.reshape(shift_labels,[-1])

    loss = tf.keras.losses.sparse_categorical_crossentropy(shift_labels, shift_logits,from_logits=True)##reduction='?'
    print(loss)
    preds = np.argmax(shift_logits,axis=1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]
    #先不考虑pad对于准确率的影响
    # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
    # not_ignore = shift_labels.ne(pad_id)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
    # num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量
    correct = (shift_labels == preds)  # 计算model预测正确的token的个数，排除pad的tokne
    print('correct={}'.format(correct))
    #correct = correct.float().sum()
    correct = 0
    for i in range(len(preds)):
        if (preds[i] == True):
            correct += 1
    accuracy = correct/len(preds)
    return loss, accuracy

BATCH_SIZE=3
def train(model,  train_list,  args):
    train_dataset = train_args.collate_fn(train_list)
    print(train_dataset)
    new_list = []
    for i in range(len(train_dataset)):
        s = list(map(int, train_dataset[i]))
        new_list.append(s)
    dd = tf.convert_to_tensor(new_list)
    train_dataset = tf.data.Dataset.from_tensor_slices(dd)
    dataset=train_dataset.batch(BATCH_SIZE,drop_remainder=True)#drop_remainder 忽略最后一个不足数量的batch
    #pytorch数据读取
    print('dataset={}'.format(dataset))
    #model.train()##model有train和evalute状态
    # # 计算所有epoch进行参数优化的总步数total_steps     ？？
    # # 设置优化器，并且在初始训练时，使用warmup策略
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    model.summary()
    checkpoint_dir='./dialogue_model'
    checkpoint_prefix=os.path.join(checkpoint_dir,"GPT2_1")
    checkpoint=tf.train.Checkpoint(optimizer=optimizer,
                                   model=model)
    #     # # 开始训练
    for epoch in range(args.epochs):
        for bantch_idx,input_ids in enumerate(dataset):
            #outputs = model(input_ids)
            print(bantch_idx)
            print('input_id={}  '.format(input_ids))
            with tf.GradientTape() as t:
                outputs = model.call(inputs=input_ids)
                loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids)
            #print('model.trainable_variables={}'.format(model.trainable_variables))
            gradients=t.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        checkpoint.save(file_prefix=checkpoint_prefix)

    #tf.saved_model.save(model, "./Model_1")
    #调用 model = tf.saved_model.load("Model_1")

def main():
    args = train_args.setup_train_args()
    if args.seed:
        train_args.set_random_seed(args)

    # 初始化tokenizer
    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    # tokenizer的字典大小
    vocab_size = len(tokenizer)
    print('vocab_size{}'.format(vocab_size))

    global pad_id
    pad_id = tokenizer.convert_tokens_to_ids(PAD)
    print('pad_id{}'.format(pad_id))

    # 创建对话模型的输出目录
    if not os.path.exists(args.dialogue_model_output_path):
        os.mkdir(args.dialogue_model_output_path)

    # 加载GPT2模型
    model, n_ctx = create_model(args, vocab_size)
    print('n_ctx{}'.format(n_ctx))

    # 对原始数据进行预处理,将原始语料转换成对应的token_id

      # 如果当前是要训练对话生成模型
    print('开始产生token')
    #preprocess_data.preprocess_raw_data(args, tokenizer, n_ctx)

    with open(args.train_tokenized_path, "r", encoding="utf8") as f:
        data = f.read()
    data_list = data.split("\n")
    result = []
    for i in range(len(data_list)):
        data = data_list[i].split(' ')
        result.append(data)
    print(result)
    data_list = result
    # return
    print('data_list{}'.format(data_list))
    train_list, test_list = train_test_split(data_list, test_size=0.1, random_state=1)


    print('开始训练')
    train(model, train_list, args)
    print('训练结束')


if __name__ == '__main__':
    main()

