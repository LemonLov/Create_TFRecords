# *-* coding:utf-8 *-*

import tensorflow as tf
import numpy as np
import os
import cv2
import random

# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 生成实数型的属性
def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def get_example_nums(tf_records_filenames):
    '''
    统计tf_records文件中图像(or example)的个数
    parameters:
        tf_records_filenames: tf_records文件路径
    return:
        nums
    '''
    nums= 0
    for record in tf.python_io.tf_record_iterator(tf_records_filenames):
        nums += 1
    return nums

def load_labels_file(filename, shuffle, labels_num=1):
    '''
    载入图片所在路径的txt文件，文件中每一行为一个图片信息，且以空格隔开：
    图像路径 标签1 标签2，如：test_image/1.jpg 0 2
    parameters:
        filename: txt文件名称
        labels_num: labels个数
        shuffle :是否打乱顺序
    return:
        images：type->list
        labels: type->list
    '''
    images = []
    labels = []
    with open(filename) as f:
        lines_list = f.readlines()
        if shuffle:
            random.shuffle(lines_list)

        for lines in lines_list:
            line = lines.rstrip().split(' ')
            label = []
            for i in range(labels_num):
                label.append(int(line[i+1]))
            images.append(line[0])
            labels.append(label)
    return images, labels

def read_image(filename, resize_height, resize_width, normalization=False):
    '''
    读取图片数据,默认返回的是uint8, [0, 255]
    parameters:
        filename: 图片路径
        resize_height: 图片的高度
        resize_width: 图片的宽度
        normalization: 是否将图片归一化至[0.0, 1.0]
    return:
        rgb_image: 返回的图片数据
    '''
    bgr_image = cv2.imread(filename)
    if len(bgr_image.shape) == 2:#若是灰度图则转为三通道
        print("Warning: gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) # 将BGR转为RGB

    if resize_height > 0 and resize_width > 0:
        rgb_image = cv2.resize(rgb_image, (resize_width, resize_height))
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        rgb_image = rgb_image / 255.0
    return rgb_image

def create_records(filename, output_record_dir, resize_height, resize_width, shuffle=True, log=20):
    '''
    实现将图像原始数据，标签，长，宽，通道数等信息保存为record文件
    注意:读取的图像数据默认是uint8，再转为tf的字符串型BytesList保存，解析请根据需要转换类型
    parameters:
        filename: 输入保存图片信息的txt文件(image_dir+filename构成图片的路径)
        output_record_dir: 保存record文件的路径
        resize_height: 图片缩放的高度
        resize_width: 图片缩放的宽度
        PS: 当resize_height或者resize_width=0时，不执行resize
        shuffle: 是否打乱顺序
        log: log信息打印间隔
    return:
        None
    '''
    # 加载文件，仅获取一个标签（一般情况下图像分类任务都是处理一个标签）
    images_list, labels_list = load_labels_file(filename, shuffle, 1)
    # 创建一个writer来写TFRecord文件
    writer = tf.python_io.TFRecordWriter(output_record_dir)
    for i, [image_name, labels] in enumerate(zip(images_list, labels_list)):
        # 构建图片相对路径
        image_path = images_list[i]
        if not os.path.exists(image_path):
            print('Error: no image path ', image_path)
            continue
        # 读取一张图片
        image = read_image(image_path, resize_height, resize_width)
        # 将图像矩阵转化为一个字符串
        image_raw = image.tostring()
        # 显示处理进程
        if i % log == 0 or i == len(images_list) - 1:
            print('------------processing: {}-th------------'.format(i))
            print('current image_path = {}'.format(image_path), 'shape: {}'.format(image.shape), 'labels: {}'.format(labels))
        # 这里仅保存一个label, 多label适当增加 'label': _int64_feature(label) 项
        label = labels[0]
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image_raw),
            'label': _int64_feature(label),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'channels': _int64_feature(image.shape[2]),
            }))
        # 将一个Example写入TFRecord文件
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    # 参数设置
    resize_height = 224  # 指定存储图片高度
    resize_width = 224  # 指定存储图片宽度
    
    # 产生train.record文件
    train_labels = './dataset/train.txt'  # 图片路径
    train_record_output = './dataset/record/train{}.tfrecords'.format(resize_height)
    create_records(train_labels, train_record_output, resize_height, resize_width)
    train_nums = get_example_nums(train_record_output)
    print("save train example nums = {}".format(train_nums))
    
    # 产生val.record文件
    val_labels = './dataset/val.txt'  # 图片路径
    val_record_output = './dataset/record/val{}.tfrecords'.format(resize_height)
    create_records(val_labels, val_record_output, resize_height, resize_width)
    val_nums = get_example_nums(val_record_output)
    print("save val example nums = {}".format(val_nums))

    # 产生test.record文件
    test_labels = './dataset/test.txt'  # 图片路径
    test_record_output = './dataset/record/test{}.tfrecords'.format(resize_height)
    create_records(test_labels, test_record_output, resize_height, resize_width)
    test_nums = get_example_nums(test_record_output)
    print("save test example nums = {}".format(test_nums))