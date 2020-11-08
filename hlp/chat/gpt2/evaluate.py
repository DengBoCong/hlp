import tensorflow as tf
import hlp.chat.gpt2.train as train
import hlp.chat.gpt2.preprocess_data as preprocess_data

BATCH_SIZE = train.BATCH_SIZE


def evaluate(model, test_list, args, tokenizer):
    test_dataset = preprocess_data.collate_fn(test_list)
    new_list = []
    for i in range(len(test_dataset)):
        s = list(map(int, test_dataset[i]))
        new_list.append(s)
    dd = tf.convert_to_tensor(new_list)
    test_dataset = tf.data.Dataset.from_tensor_slices(dd)
    dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)  # drop_remainder

    for batch_idx, input_ids in enumerate(dataset):
        outputs = model.call(input_ids=input_ids)
        loss, accuracy = train.calculate_loss_and_accuracy(outputs, labels=input_ids, tokenizer=tokenizer)

        if args.gradient_accumulation > 1:
            loss = loss / args.gradient_accumulation
            accuracy = accuracy / args.gradient_accumulation
