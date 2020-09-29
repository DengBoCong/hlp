import logging
import os
import time
import numpy as np
import pytest
import tensorflow as tf
from tacotron2.dataset.dataset_wav import Dataset_wave
from tacotron2.dataset.dataset_txt import Dataset_txt
from tacotron2.config.config import Tacotron2Config
from tacotron2.model.tacotron2 import TFTacotron2
from tacotron2.plot.plot import plot_mel
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logging.basicConfig(
     level=logging.WARNING,
     format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
 )


@pytest.mark.parametrize(
    "n_speakers, n_chars, max_input_length, max_mel_length, batch_size",
    [(1, 15, 84, 500, 2),],
)
def test_tacotron2_trainable(
    n_speakers, n_chars, max_input_length, max_mel_length, batch_size
):
    config = Tacotron2Config(n_speakers=n_speakers, reduction_factor=1)
    model = TFTacotron2(config, training=True)

    path_to_file = r".\text\文字.txt"
    input_ids = Dataset_txt(path_to_file)
    input_ids=tf.convert_to_tensor(input_ids)
    speaker_ids = tf.convert_to_tensor([0] * batch_size, tf.int32)
    path = r".\wavs1"
    mel_gts = Dataset_wave(path)
    mel_gts = tf.cast(mel_gts, tf.float32)
    plot_mel(mel_gts)

    mel_lengths = np.random.randint(
        max_mel_length, high=max_mel_length + 1, size=[batch_size]
    )

    mel_lengths[-1] = max_mel_length
    mel_lengths = tf.convert_to_tensor(mel_lengths, dtype=tf.int32)

    optimizer = tf.keras.optimizers.Adam(lr=0.001)

    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function(experimental_relax_shapes=True)
    def one_step_training(input_ids, speaker_ids, mel_gts, mel_lengths):
        with tf.GradientTape() as tape:
            mel_preds, post_mel_preds, stop_preds, alignment_history = model(
                input_ids,
                tf.constant([max_input_length, max_input_length]),
                speaker_ids,
                mel_gts,
                mel_lengths,
                training=True,
            )
            loss_before = tf.keras.losses.MeanSquaredError()(mel_gts, mel_preds)
            loss_after = tf.keras.losses.MeanSquaredError()(mel_gts, post_mel_preds)

            stop_gts = tf.expand_dims(
                tf.range(tf.reduce_max(mel_lengths), dtype=tf.int32), 0
            )  # [1, max_len]
            stop_gts = tf.tile(stop_gts, [tf.shape(mel_lengths)[0], 1])  # [B, max_len]
            stop_gts = tf.cast(
                tf.math.greater_equal(stop_gts, tf.expand_dims(mel_lengths, 1) - 1),
                tf.float32,
            )

            # calculate stop_token loss
            stop_token_loss = binary_crossentropy(stop_gts, stop_preds)

            loss = stop_token_loss + loss_before + loss_after

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, alignment_history,post_mel_preds

    for i in range(2):
        if i == 1:
            start = time.time()
        loss, alignment_history,post_mel_preds = one_step_training(
            input_ids, speaker_ids, mel_gts, mel_lengths
        )
        print(f" > loss: {loss}")
    total_runtime = time.time() - start
    print(f" > Total run-time: {total_runtime}")
    print(f" > Avg run-time: {total_runtime/10}")
    plot_mel(post_mel_preds)
