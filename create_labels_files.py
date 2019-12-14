# -*- coding: utf-8 -*-

import os

def write_txt(file_list, filename, mode='w'):
    """
    保存图片路径+标签信息到txt文件中
    parameters:
        file_list: 需要保存的图片路径+标签,type->list
        filename: txt文件名
        mode: 读写模式: 'w' or 'a'
    return:
        None
    """
    with open(filename, mode) as f:
        for line in file_list:
            str_line = ""
            # print(line)
            for col, data in enumerate(line):
                # print(col)
                if not col == len(line) - 1:
                    # 每行中当不是最后一个数据时，以空格作为分隔符
                    str_line = str_line + str(data) + " "
                else:
                    # 每行最后一个数据用换行符“\n”
                    str_line = str_line + str(data) + "\n"
            # print(str_line)
            f.write(str_line)

def get_files_list(dir):
    '''
    实现遍历dir目录下,所有文件夹中(包含子文件夹)的所有文件
    parameters:
        dir: 指定文件夹目录
    return:
        files_list: 包含所有文件路径和标签信息的列表
    '''
    # parent:父目录, dirnames:该目录下所有文件夹, filenames:该目录下的所有文件名
    files_list = []
    current_label = 0
    for parent, dirnames, filenames in os.walk(dir):
        # print("parent is: " + parent)
        # print("dirnames is: ", dirnames)
        # print("filenames is: ", filenames)
        if not filenames:
            continue
        for filename in filenames:
            # print("filename is: " + filename)
            files_list.append([os.path.join(parent, filename), current_label])
        current_label += 1
    return files_list


if __name__ == '__main__':
    train_dir = './dataset/train'
    train_txt = './dataset/train.txt'
    train_data_list = get_files_list(train_dir)
    write_txt(train_data_list, train_txt, mode='w')

    val_dir = './dataset/val'
    val_txt = './dataset/val.txt'
    val_data_list = get_files_list(val_dir)
    write_txt(val_data_list, val_txt, mode='w')

    test_dir = './dataset/test'
    test_txt = './dataset/test.txt'
    test_data_list = get_files_list(test_dir)
    write_txt(test_data_list, test_txt, mode='w')
