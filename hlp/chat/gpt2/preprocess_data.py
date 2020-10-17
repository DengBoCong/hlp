from tqdm import tqdm #可以在控制台显示进度条


def preprocess_raw_data(args, tokenizer, n_ctx):
    """
    对原始语料进行处理，将原始语料转换为用于train的token id，对于每个dialogue，将其处于成如下形式"[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    :param args:
    :param tokenizer:
    :param n_ctx:GPT2模型的上下文窗口大小,对于超过n_ctx(n_ctx包括了特殊字符)的dialogue进行截断
    :return:
    """
    with open(args.train_raw_path, 'rb') as f:
        data = f.read().decode("utf-8")##整体未切分的数据
        ##数据的结构是 一段话最后一段话的回复不同 这样的前同后不同 会持续好几个 大概 6次吧
    if "\r\n" in data:
        train_data = data.split("\r\n\r\n")###切成一段一段的
    else:
        train_data = data.split("\n\n")
    i=0
    with open(args.train_tokenized_path, "w", encoding="utf-8") as f:
        for dialogue_index, dialogue in enumerate(tqdm(train_data)): ##每段话进行循环 抽取
            if "\r\n" in data:   ##切分句子 成一句一句的
                utterances = dialogue.split("\r\n")
            else:
                utterances = dialogue.split("\n")
            dialogue_ids = [tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
            ##完成一段话的处理 生成dialogue_ids
            for utterance in utterances:  ##读取每一句话 把话以字为单位转化为token 然后每句之间 sep为间隔  直到一段话读完 then 这段话合并成了一句
                dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in utterance])
                dialogue_ids.append(tokenizer.sep_token_id)  # #每个utterance之后添加[SEP]，表示utterance结束
            # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
            dialogue_ids = dialogue_ids[:n_ctx]  #自设 n_ctx---统一长度
            #print('dialogue_ids={}'.format(dialogue_ids))
            #print('n_ctx={}'.format(n_ctx))
            ##将处理好的token写入文件
            for dialogue_id in dialogue_ids:  #对一段话中的每个字id 转换成str 后加 空格 隔开
                f.write(str(dialogue_id) + ' ')
            # 最后一条记录不添加换行符----避免空行的产生
            if dialogue_index < len(train_data) - 1:
                f.write("\n")
    f.close()


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
    return input_ids, max_input_len