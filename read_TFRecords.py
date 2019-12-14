# -*-coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

def show_image(title, image):
    '''
    显示图片
    parameters:
        title: 图像标题
        image: 图像数据
    return:
        None
    '''
    plt.figure()
    plt.imshow(image)
    plt.axis('on')    # 关掉坐标轴为 off
    plt.title(title)  # 图像标题
    plt.show()

def read_records(filename, resize_height, resize_width):
    '''
    解析record文件:源文件的图像数据是RGB, uint8, [0, 255]
    parameters:
        filename:
        resize_height:
        resize_width:
    return:
        None
    '''
    # 创建输入文件队列
    filename_queue = tf.train.string_input_producer([filename], shuffle=False)
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channels': tf.FixedLenFeature([], tf.int64),
        }) 

    image, label = features['image'], tf.cast(features['label'], tf.int32)
    height, width = features['height'], features['width']
    channels = features['channels']
    
    # 从原始图像数据中解析出像素矩阵，并根据图像尺寸还原图像
    decoded_image = tf.decode_raw(image, tf.uint8)
    # 设置图像的维度
    decoded_image = tf.reshape(decoded_image, [resize_height, resize_width, 3])
    # 转换图像张量的类型
    decoded_image = tf.image.convert_image_dtype(decoded_image, dtype=tf.float32)

    # 这里仅仅返回图像和标签
    return decoded_image, label

def disp_records(record_file, resize_height, resize_width, show_nums=4):
    '''
    解析record文件，并显示show_nums张图片，主要用于验证是否成功生成record文件
    parameters:
        record_file: record文件路径
        resize_height:
        resize_width:
        show_nums: 展示多少张图片
    return:
        None
    '''
    # 读取record函数
    image, label = read_records(record_file, resize_height, resize_width)
    image = preprocess_for_train(image, resize_height, resize_width, None)
    # 显示前4个图片
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(show_nums):
            tf_image, tf_label = sess.run([image, label])  # 在会话中取出image和label
            print('shape: {}, tpye: {}, labels: {}'.format(tf_image.shape, tf_image.dtype, tf_label))
            # print(tf_image)
            show_image("image label: {}".format(tf_label), tf_image)

        # 停止所有线程
        coord.request_stop()
        coord.join(threads)

def distort_color(image, color_ordering=0):
    """
    随机调整图像的色彩，包括亮度、对比度、饱和度和色相，可以定义多种调整顺序。
    parameters:
        image: 输入图像
        color_ordering: 调整顺序
    return:
        distort_color_image: 调整过色彩后的图像
    """
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    distort_color_image = tf.clip_by_value(image, 0.0, 1.0)
    return distort_color_image

def preprocess_for_train(image, height, width, bbox):
    """
    parameters:
        image: 输入图像
        height: 图像的高度
        width: 图像的宽度
        bbox: 标注框
    return:
        distorted_image: 预处理后的图像
    """
    if bbox is None:
        # 如果没有提供标注框， 则认为整个图像就是需要关注的部分
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    # 转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # 随机截取图像
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    # 调整图像大小
    distorted_image = tf.image.resize_images(image, [height, width], method=np.random.randint(4))
    # 随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 使用随机顺序调整图像色彩
    distorted_image = distort_color(distorted_image, np.random.randint(3))
    # 设置图像的维度
    distorted_image = tf.reshape(distorted_image, [height, width, 3])
    return distorted_image

def get_batch_images(images, labels, batch_size, labels_nums, one_hot=True, shuffle=True, num_threads=1):
    '''
    获取一个batch大小的数据
    parameters:
        images: 图像
        labels: 标签
        batch_size: 一个batch的大小
        labels_nums: 标签个数
        one_hot: 是否将labels转为one_hot的形式
        shuffle: 是否打乱顺序，一般train时shuffle=True，验证时shuffle=False
    return:
        images_batch, labels_batch: 返回一个batch的images和labels
    '''
    min_after_dequeue = 300
    capacity = min_after_dequeue + 3 * batch_size  # 保证capacity必须大于min_after_dequeue参数值
    if shuffle:
        images_batch, labels_batch = tf.train.shuffle_batch([images,labels],
                                                            batch_size=batch_size,
                                                            capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue,
                                                            num_threads=num_threads)
    else:
        images_batch, labels_batch = tf.train.batch([images,labels],
                                                    batch_size=batch_size,
                                                    capacity=capacity,
                                                    num_threads=num_threads)
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, labels_nums, 1, 0)
    return images_batch, labels_batch

def batch_test(record_file, resize_height, resize_width, batch=5):
    '''
    parameters:
        record_file: record文件路径
        resize_height:
        resize_width:
        batch: 测试几个batch
    return:
        None
    PS:image_batch, label_batch一般作为网络的输入
    '''
    # 读取record函数
    image, label = read_records(record_file, resize_height, resize_width)
    image = preprocess_for_train(image, resize_height, resize_width, None)
    image_batch, label_batch = get_batch_images(image, label, batch_size=4, labels_nums=5, one_hot=False)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(batch):
            # 在会话中取出images和labels
            images, labels = sess.run([image_batch, label_batch])
            # 这里仅显示每个batch里第一张图片
            print('shape: {}, tpye: {}, labels: {}'.format(images.shape, images.dtype, labels))
            # print(images)
            show_image("{} of {} batch's image , label: {}".format(1, i, labels[0]), images[0, :, :, :])

        # 停止所有线程
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    
    # 参数设置
    resize_height = 224  # 存储图片的高度
    resize_width = 224  # 存储图片的宽度

    # train.record文件
    train_record_output = 'dataset/record/train{}.tfrecords'.format(resize_height)
    
    # val.record文件
    val_record_output = 'dataset/record/val{}.tfrecords'.format(resize_height)
    
    # 测试显示函数
    disp_records(train_record_output, resize_height, resize_width)
    batch_test(train_record_output, resize_height, resize_width)
