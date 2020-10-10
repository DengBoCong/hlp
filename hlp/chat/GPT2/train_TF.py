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
import train_args
import predict_arg
import preprocess_data
#数据处理
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
def calculate_loss_and_accuracy(outputs, labels,tokenizer):
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
    shift_labels=tf.convert_to_tensor(shift_labels)
    shift_logits=tf.reshape(shift_logits,[-1, tf.convert_to_tensor(shift_logits).shape[-1]])
    shift_labels=tf.reshape(shift_labels,[-1])
    #shift_logits是预测的各个单词出现的概率
    print('预测结果={}'.format(shift_logits))
    print('shift_logits.shape={}'.format(shift_logits))
    loss = tf.keras.losses.sparse_categorical_crossentropy(shift_labels, shift_logits,from_logits=True)##reduction='?'

    preds = np.argmax(shift_logits,axis=1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]
    print('预测id={}'.format(preds))
    print('预测label={}'.format(shift_labels))
    #先不考虑pad对于准确率的影响
    # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
    # not_ignore = shift_labels.ne(pad_id)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
    # num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量
    correct = 0  # 计算model预测正确的token的个数，排除pad的tokne
    #correct = correct.float().sum()
    print('逐字打印出 预测的值 和 对应的label值')
    text = tokenizer.convert_ids_to_tokens(preds)
    print('预测的文本序列：={}'.format(text))

    for i in range(len(preds)):
        #text = tokenizer.convert_ids_to_tokens(preds[i])
        # print('text={}'.format(text))
        # print('shift_labels={}'.format(shift_labels[i]))
        if (preds[i] == shift_labels[i]):
            correct += 1
    print('correct={}'.format(correct))
    accuracy = correct/len(preds)
    return loss, accuracy

BATCH_SIZE=3
def train(model,  train_list,  args,tokenizer):
    train_dataset = train_args.collate_fn(train_list)
    print(train_dataset)
    new_list = []
    for i in range(len(train_dataset)):
        s = list(map(int, train_dataset[i]))
        new_list.append(s)
    dd = tf.convert_to_tensor(new_list)
    train_dataset = tf.data.Dataset.from_tensor_slices(dd)
    dataset=train_dataset.batch(BATCH_SIZE,drop_remainder=True)#drop_remainder 忽略最后一个不足数量的batch
    #数据读取
    print('dataset={}'.format(dataset))
    #model.train()##model有train和evalute状态
    # # 计算所有epoch进行参数优化的总步数total_steps     ？？
    # # 设置优化器，并且在初始训练时，使用warmup策略
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    model.summary()

    #     # # 开始训练
    for epoch in range(args.epochs):
        print('epoch={}'.format(epoch))
        for bantch_idx,input_ids in enumerate(dataset):
            #outputs = model(input_ids)
            #print('bantch_idx={}'.format(bantch_idx))
            #print('input_id={}  '.format(input_ids))
            with tf.GradientTape() as t:
                outputs = model.call(inputs=input_ids)
                loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids,tokenizer=tokenizer)
            #print('model.trainable_variables={}'.format(model.trainable_variables))
            gradients=t.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print('epoch={} loss={} accuracy={} '.format(epoch,loss,accuracy))
    model.save_weights('model_weight')
    return model

    # model_name="Model_1/model"
    # model.save('saved_model/my_model')
    #model.save('Model_1/my_model')#此方法适用于功能模型或者顺序模型 不适用于此种子类模型
    # 调用 model = tf.saved_model.load("Model_1")


def predict(model):

    args = predict_arg.set_interact_args()
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))

    logger = predict_arg.create_logger(args)
    tokenizer = BertTokenizer(vocab_file=args.voca_path)
    vocab_size = len(tokenizer)

    # args_train = train_args.setup_train_args()
    # model, _ = train_TF.create_model(args_train, vocab_size)
    # model.load_weights('model_weight')

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
                filtered_logits,promax_index = predict_arg.top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                #tf.raw_ops.Multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标

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
            print('generated={}'.format(generated))
            #print("history:{}".format(history))
            text = tokenizer.convert_ids_to_tokens(generated)
            print("chatbot:" + "".join(text))
            if args.save_samples_path:
                samples_file.write("chatbot:{}\n".format("".join(text)))
        except KeyboardInterrupt:
            if args.save_samples_path:
                samples_file.close()
            break

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
    print('n_ctx={}'.format(n_ctx))
    print('kkkk')

    # 对原始数据进行预处理,将原始语料转换成对应的token_id

      # 如果当前是要训练对话生成模型
    print('开始产生token')#修改了数据集的话  要取消掉此注释 因为数据集每行最后的空格问题
    #现在的解决方法就是 每次都手动删除最后一个空格 然后 生成token之后 要注释掉下面一行
    #preprocess_data.preprocess_raw_data(args, tokenizer, n_ctx)

    with open(args.train_tokenized_path, "r", encoding="utf8") as f:
        data = f.read()
    data_list = data.split("\n")
    result = []
    for i in range(len(data_list)):
        data = data_list[i].split(' ')
        result.append(data)
    #print(result)
    data_list = result
    # return
    print('data_list{}'.format(data_list))

    train_list, test_list = train_test_split(data_list, test_size=0.1, random_state=1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    checkpoint_dir = './dialogue_model'
    checkpoint_prefix = os.path.join(checkpoint_dir, "GPT2_1")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                      model=model)


    print('开始训练')
    model=train(model, train_list, args,tokenizer)

    checkpoint.save(file_prefix=checkpoint_prefix)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    print('训练结束')
    print('开始交互')
    predict(model)

if __name__ == '__main__':
    main()

