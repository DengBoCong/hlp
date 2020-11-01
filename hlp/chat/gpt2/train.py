import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import train_args as train_args
from gpt2 import TFGPT2Model
from gpt2_config import GPT2Config
import poem_proprocess_raw_data as preprocess_data
import time

# 数据处理
PAD = '[PAD]'
pad_id = 0


def create_model(args, config):
    """
    :param args:
    :return:
    """

    print('创建model')
    model = TFGPT2Model(config)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=3e-5,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9)

    return model, config.n_ctx, optimizer


def change_tpye(outputs, labels):
    logits = outputs[0]  # (batch,len,vocab)

    shift_labels = labels[:, 1:]  # (batch,len,vocab)
    output_logits = logits[:, :-1, ]  # (batch,len,vocab)

    shift_labels = tf.convert_to_tensor(shift_labels)
    shift_logits = tf.reshape(output_logits, [-1, tf.convert_to_tensor(output_logits).shape[-1]])  # (batch*len,vocab)
    shift_labels = tf.reshape(shift_labels, [-1])  # (batch*len,)

    return shift_logits, shift_labels, output_logits


def loss_function(shift_logits, shift_labels, tokenizer, output_logits, train_loss, train_accuracy):
    mask = tf.math.logical_not(tf.math.equal(shift_labels, 0))
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    loss_ = loss_object(shift_labels, shift_logits)  # 一维 len*bantch
    mask = tf.cast(mask, dtype=loss_.dtype)  # 一维 len*bantch
    loss_ *= mask
    loss = tf.reduce_mean(loss_)
    train_loss(loss)
    train_accuracy(shift_labels, shift_logits)
    # preds = np.argmax(output_logits[0],
    #                   axis=1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]
    #
    # correct = 0  # 计算model预测正确的token的个数，排除pad的tokne
    # text = tokenizer.convert_ids_to_tokens(preds)
    # for i in range(len(preds)):
    #     # text = tokenizer.convert_ids_to_tokens(preds[i])
    #     # print('text={}'.format(text))
    #     # print('shift_labels={}'.format(shift_labels[i]))
    #     if (preds[i] == shift_labels[i]):
    #         correct += 1
    # accuracy = correct / len(preds)
    return loss, train_loss, train_accuracy


def load_checkpoint(model, optimizer, args):
    # 加载检查点
    checkpoint_path = args.dialogue_model_output_path
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('已恢复至最新的检查点！')


def train_step(model, input_ids, optimizer, tokenizer, train_loss, train_accuracy):
    with tf.GradientTape() as t:
        outputs = model(
            inputs=input_ids)  # input_ids  (bantch_size,dim)   outputs : [(batch_size, dim, vocab),(batch,2,head,4,dim,32)]  第二个？
        shift_logits, shift_labels, output_logits = change_tpye(outputs, input_ids)
        # shift_logits:[batch*(dim-1),vocab]; shift_labels:[(dim-1),] ; output_logits：(batch_size, dim-1, vocab)
        loss, train_loss, train_accuracy = loss_function(shift_logits, shift_labels, tokenizer, output_logits,
                                                         train_loss, train_accuracy)
    gradients = t.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return train_loss, train_accuracy


def train(model, args, tokenizer, optimizer, train_loss, train_accuracy):
    train_list, dataset, max_input_len = data_process(args)
    # 数据读取
    checkpoint_path = args.dialogue_model_output_path

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('已恢复至最新检查点！')
    print("开始训练...")

    # # 开始训练
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        sample_sum = int(args.data_size * 0.9)
        batch_sum = 0
        for batch_idx, input_ids in enumerate(dataset):
            train_loss, train_accuracy = train_step(model, input_ids, optimizer, tokenizer, train_loss, train_accuracy)
            batch_sum = batch_sum + len(input_ids)
            print('\r', '[Batch {} Loss {:.4f} Accuracy {:.4f}]'.format(batch_idx, train_loss.result(),
                                                                        train_accuracy.result()), end='', flush=True)
        epoch_time = (time.time() - start)
        step_time = epoch_time * args.batch_size / sample_sum
        print(' - {:.0f}s - {:.0f}ms/step - loss: {:.4f} - Accuracy {:.4f}'.format(epoch_time, step_time * 1000
                                                                                   , train_loss.result()
                                                                                   , train_accuracy.result()))
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))


def data_process(args, ):
    # 进行数据类型变换
    with open(args.train_tokenized_path, "r", encoding="utf8") as f:
        data_list = []
        # 一行行地读取 str类型的data  然后转换为list形式
        # data_list  最终形状 [  [],[],……,[]  ]
        for line in f.readlines():
            data = line.strip()
            data = data.split(' ')
            data_list.append(data)

    train_list, test_list = train_test_split(data_list, test_size=0.1, random_state=1)
    train_dataset, max_input_len = preprocess_data.collate_fn(train_list)
    new_list = []
    for i in range(len(train_dataset)):
        s = list(map(int, train_dataset[i]))
        new_list.append(s)
    dd = tf.convert_to_tensor(new_list)
    train_dataset = tf.data.Dataset.from_tensor_slices(dd)
    dataset = train_dataset.batch(args.batch_size,
                                  drop_remainder=True)  # drop_remainder 忽略最后一个不足数量的batch  #[batchsize, ? ]  32,64
    return train_list, dataset, max_input_len


def main():
    args = train_args.setup_train_args()
    if args.seed:
        train_args.set_random_seed(args)
    # 初始化tokenizer
    tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    # tokenizer的字典大小
    global pad_id
    # pad_id = tokenizer.convert_tokens_to_ids(PAD)

    # 创建对话模型的输出目录
    if not os.path.exists(args.dialogue_model_output_path):
        os.mkdir(args.dialogue_model_output_path)

    # 加载GPT2模型
    config = GPT2Config()
    model, n_ctx, optimizer = create_model(args, config)

    # 对原始数据进行预处理,将原始语料转换成对应的token_id
    # 如果当前是要训练对话生成模型
    print('开始产生token')
    # 不修改数据集的情况下，没必要每次训练都运行preprocess_raw_data 因为 生成的data是一样的
    if not os.path.exists(args.train_tokenized_path):
        file = open(args.train_tokenized_path, 'w')

    preprocess_data.preprocess_raw_data(args, tokenizer, n_ctx)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    print('开始训练')
    train(model, args, tokenizer, optimizer, train_loss, train_accuracy)
    print('训练结束')


if __name__ == '__main__':
    main()
