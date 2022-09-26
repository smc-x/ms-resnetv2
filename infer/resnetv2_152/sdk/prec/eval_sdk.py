""" eval_sdk.py """
import os
import numpy as np


def read_file_list(input_f):
    """
    :param infer file content:
        1.bin 0
        2.bin 2
        ...
    :return image path list, label list
    """
    image_file_l = []
    label_l = []
    if not os.path.exists(input_f):
        print('input file does not exists.')
    with open(input_f, "r") as fs:
        for line in fs.readlines():
            line = line.strip('\n').split(',')
            file = line[0]
            label = int(line[1])
            image_file_l.append(file)
            label_l.append(label)
    return image_file_l, label_l
# path to result and label
images_txt_path = "../../data/image/cifar/label.txt"
infer_results_txt = '../result/infer_results.txt'
# load results and label
results = np.loadtxt(infer_results_txt)
file_list, label_list = read_file_list(images_txt_path)
img_size = len(file_list)

labels = np.array(label_list)
# cal acc
acc_top1 = (results[:, 0] == labels).sum() / img_size
print('Top1 acc:', acc_top1)

acc_top5 = sum([1 for i in range(img_size) if labels[i] in results[i]]) / img_size
print('Top5 acc:', acc_top5)
