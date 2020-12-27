import os

import numpy as np
import tensorflow as tf


def load_checkpoint(model: tf.keras.Model, checkpoint_dir: str, checkpoint_save_size: int):
    """
    恢复检查点
    """
    # 如果检查点存在就恢复，如果不存在就重新创建一个
    checkpoint = tf.train.Checkpoint(wavernn=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=checkpoint_save_size)

    if os.path.exists(checkpoint_dir):
        if ckpt_manager.latest_checkpoint:
            checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # if execute_type == "generate":
        #     print("没有检查点，请先执行train模式")
        #     exit(0)

    return ckpt_manager


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    dim = len(x.shape) - 1
    m = tf.reduce_max(x, axis=dim)
    m2, _ = tf.reduce_max(x, axis=dim, keepdims=True)
    return m + tf.math.log(tf.reduce_sum(tf.exp(x - m2), axis=dim))


# It is adapted from https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/mixture.py
def Discretized_Mix_Logistic_Loss(y_hat, y, num_classes=65536,
                                  log_scale_min=None, reduce=True):
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    y_hat = tf.transpose(y_hat, (0, 2, 1))

    # assert y_hat.dim() == 3
    assert y_hat.shape[1] % 3 == 0
    nr_mix = y_hat.shape[1] // 3

    # (B x T x C)
    y_hat = tf.transpose(y_hat, (0, 2, 1))

    # unpack parameters. (B, T, num_mixtures) x 3
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:2 * nr_mix]
    log_scales = tf.clip_by_value(y_hat[:, :, 2 * nr_mix:3 * nr_mix], clip_value_min=log_scale_min,
                                  clip_value_max=10000000)

    # B x T x 1 -> B x T x num_mixtures
    y = tf.tile(y, (1, 1, means.shape[-1]))
    centered_y = y - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = tf.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = tf.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(F.sigmoid(plus_in))
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - F.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)

    # tf equivalent
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
    # for num_classes=65536 case? 1e-7? not sure..
    inner_inner_cond = tf.cast((cdf_delta > 1e-5), dtype=float)

    inner_inner_out = inner_inner_cond * \
                      tf.math.log(tf.clip_by_value(cdf_delta, clip_value_min=1e-12, clip_value_max=100000000)) + \
                      (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
    inner_cond = tf.cast((y > 0.999), dtype=float)
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = tf.cast((y < -0.999), dtype=float)
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    log_probs = log_probs + tf.nn.log_softmax(logit_probs, -1)

    if reduce:
        return -tf.reduce_mean(log_sum_exp(log_probs))
    else:
        return -tf.expand_dims(log_sum_exp(log_probs), axis=-1)
