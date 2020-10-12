import tensorflow as tf
import transformers
from transformers import GPT2Config
import os
import numpy as np
from tqdm import tqdm #可以在控制台显示进度条
from sklearn.model_selection import train_test_split
from transformers import  TFGPT2LMHeadModel
from transformers import BertTokenizer
#不能是一一对应 得是理解了之后的重新编写
import hlp.chat.gpt2.train_args as train_args
import hlp.chat.gpt2.preprocess_data as preprocess_data
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
    print(vocab_size)
    print('创建model')
    #model = TFGPT2LMHeadModel.from_pretrained('gpt2')
    if args.pretrained_model:  # 如果指定了预训练的GPT2模型
        model = TFGPT2LMHeadModel.from_pretrained(args.pretrained_model)
    else:  # 若没有指定预训练模型，则初始化模型
        print('初始化模型')
        model_config = GPT2Config.from_json_file(args.model_config)
        print('config:\n' + model_config.to_json_string())
        model = TFGPT2LMHeadModel(config=model_config)
        print('构造好模型')
        # 根据tokenizer的vocabulary调整GPT2模型的voca的大小
    #model.resize_token_embeddings(vocab_size)

    #model = TFGPT2LMHeadModel.from_pretrained()#实例化一个类
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

    # # 计算所有epoch进行参数优化的总步数total_steps     ？？
    # # 设置优化器，并且在初始训练时，使用warmup策略
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    # # 开始训练
    for epoch in range(args.epochs):
        print('epoch={}'.format(epoch))
        for bantch_idx,input_ids in enumerate(dataset):
            with tf.GradientTape() as t:
                outputs = model.call(inputs=input_ids)
                loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids,tokenizer=tokenizer)
            #print('model.trainable_variables={}'.format(model.trainable_variables))
            gradients=t.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print('epoch={} loss={} accuracy={} '.format(epoch,loss,accuracy))
    model.save_weights('./dialogue_model/model_weight')
    #model.save('Model_1/my_model')#此方法适用于功能模型或者顺序模型 不适用于此种子类化模型
    # 调用 model = tf.saved_model.load("Model_1")

def evaluate(model, test_list,  args, tokenizer):

    # 记录tensorboardX
    test_dataset = train_args.collate_fn(test_list)
    print(test_dataset)
    new_list = []
    for i in range(len(test_dataset)):
        s = list(map(int, test_dataset[i]))
        new_list.append(s)
    dd = tf.convert_to_tensor(new_list)
    test_dataset = tf.data.Dataset.from_tensor_slices(dd)
    dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)  # drop_remainder

    for batch_idx, input_ids in enumerate(dataset):
        outputs = model.call(input_ids=input_ids)
        loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, tokenizer=tokenizer)

        if args.gradient_accumulation > 1:
            loss = loss / args.gradient_accumulation
            accuracy = accuracy / args.gradient_accumulation
def main():
    args = train_args.setup_train_args()
    if args.seed:
        train_args.set_random_seed(args)
    # 初始化tokenizer
    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    # tokenizer的字典大小
    vocab_size = len(tokenizer)
    #print('vocab_size{}'.format(vocab_size))

    global pad_id
    pad_id = tokenizer.convert_tokens_to_ids(PAD)
    #print('pad_id{}'.format(pad_id))

    # 创建对话模型的输出目录
    if not os.path.exists(args.dialogue_model_output_path):
        os.mkdir(args.dialogue_model_output_path)

    # 加载GPT2模型
    model, n_ctx = create_model(args, vocab_size)
    #print('n_ctx={}'.format(n_ctx))
    # 对原始数据进行预处理,将原始语料转换成对应的token_id
      # 如果当前是要训练对话生成模型
    print('开始产生token')
    #不修改数据集的情况下，没必要每次训练都运行preprocess_raw_data 因为 生成的data是一样的
    preprocess_data.preprocess_raw_data(args, tokenizer, n_ctx)
    #进行数据类型变换
    with open(args.train_tokenized_path, "r", encoding="utf8") as f:
        data_list = []
        # 一行行地读取 str类型的data  然后转换为list形式
        for line in f.readlines():
            data = line.strip()
            data = data.split(' ')
            data_list.append(data)

    train_list, test_list = train_test_split(data_list, test_size=0.1, random_state=1)

    print('开始训练')
    train(model, train_list, args,tokenizer)
    evaluate(model, test_list, args,tokenizer)
    print('训练结束')

if __name__ == '__main__':
    main()

