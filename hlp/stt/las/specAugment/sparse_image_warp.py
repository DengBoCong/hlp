# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:51:17 2020
sparse_image_warp函数的内部是如何执行的
@author: 九童
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow_addons.image import sparse_image_warp
from hlp.stt.las.specAugment import spec_augment
import numpy as np
import tensorflow as tf



if __name__ == "__main__":
    #arr = np.random.rand(1,3,4,1,seed = 1)
    np.random.seed(0)
    # reshape spectrogram shape to [batch_size, time, frequency, 1]

    sss = np.random.random((1,8,6,1))
    
    spec_augment.visualization_spectrogram(sss,"hhh")

    print("sss:\n{}".format(sss))#(1, 8, 6, 1)

    time_warping_para = 2
    fbank_size = tf.shape(sss)
    n, v = fbank_size[1], fbank_size[2]
    print("n: {}".format(n))
    print("v: {}".format(v))
    print("time_warping_para: {}".format(time_warping_para))
    """
    n: 256
    v: 92
    time_warping_para: 80
    pt: 105
    """
    #n为该频谱图的时间步
    #v为该频谱图的频率
    # 步骤1 : 时间扭曲
    # 图像扭曲控制点设置。
    # 源
    pt = tf.random.uniform([], time_warping_para, n - time_warping_para, tf.int32)  # radnom point along the time axis
    print("pt: {}".format(pt))#(80,176)之间的一个随机数
    src_ctr_pt_freq = tf.range(v // 2)  # [0,46)的一系列数字control points on freq-axis

    src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt  # 返回一个全为1，形状相同的张量，control points on time-axis

    src_ctr_pts = tf.stack((src_ctr_pt_freq, src_ctr_pt_time), -1)

    src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)

    w = 20
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    #dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq), -1)
    dest_ctr_pts = tf.stack((dest_ctr_pt_freq, dest_ctr_pt_time), -1)
    dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)
    #print("dest_ctr_pts: {}".format(dest_ctr_pts)) 
    # 扭曲
    source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
    dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0) 
    
    print("source_control_point_locations:\n{}".format(source_control_point_locations))#(1, 3, 2)
    print("dest_control_point_locations:\n{}".format(dest_control_point_locations))#(1, 3, 2)
     
    
    
    
    
    
    
    warped_image, _ = sparse_image_warp(sss,
                                        source_control_point_locations,
                                        dest_control_point_locations)
    print("warped_image:\n{}".format(warped_image))#(1, 8, 6, 1)

    spec_augment.visualization_spectrogram(warped_image,"sss")