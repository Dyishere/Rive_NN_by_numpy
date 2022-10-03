# -*- coding: utf-8 -*-
# @Time     : 2022/10/3 17:29
# @Author   : Dyishere
# @File     : data_process.py
"""
功能描述：

"""
import os
import cv2


def draw(image_list, src_img_dir=None, tar_img_dir=None):
    if not os.path.exists(tar_img_dir):
        os.mkdir(tar_img_dir)
    for item in image_list:
        sub_path = item["path"]
        path_seg = sub_path.split("/")
        path = os.path.join(src_img_dir, sub_path)
        boxes = item["boxes"]
        img = cv2.imread(path)
        for box in boxes:
            ord = box.split(" ")
            x, y, w, h = int(ord[0]), int(ord[1]), int(ord[2]), int(ord[3])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        tar_dir = os.path.join(tar_img_dir, path_seg[0])
        if not os.path.exists(tar_dir):
            os.mkdir(tar_dir)
        tar_path = os.path.join(tar_dir, path_seg[1])
        cv2.imwrite(tar_path, img)


def parse(label_file_path, src_img_dir, tar_img_dir):
    fr = open(label_file_path, 'r')
    assert fr is not None
    image_list = []
    line = fr.readline().rstrip()
    # print("line:{}".format(line))
    while line:
        mdict = {}
        path = line
        mdict["path"] = path
        # print("path:{}".format(path))
        num = fr.readline().rstrip()
        # print("num:{}".format(num))
        boxes_list = []
        for n in range(int(num)):
            box = fr.readline().rstrip()
            boxes_list.append(box)
        mdict["boxes"] = boxes_list
        image_list.append(mdict)
        line = fr.readline().rstrip()
    print(image_list[0])
    draw(image_list, src_img_dir, tar_img_dir)


if __name__ == '__main__':
    file_path = "D:/Dataset/wider/wider_face_split/wider_face_train_bbx_gt.txt"
    source_img_dir = "D:/Dataset/wider/WIDER_train/images"
    target_img_dir = "D:/Dataset/wider/WIDER_train/drew"
    parse(file_path, source_img_dir, target_img_dir)
