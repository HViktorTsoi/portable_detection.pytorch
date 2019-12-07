import os
import random

import os.path as osp
import pandas as pd
from PIL import Image

import torchvision

voc_annotation_template = """
<annotation>
    <folder>{}</folder>
    <filename>{}</filename>
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>{}</depth>
    </size>
    {}
    <segmented>0</segmented>
</annotation>
"""
voc_object_template = """
    <object>
        <name>{}</name>
        <bndbox>
            <xmin>{}</xmin>
            <xmax>{}</xmax>
            <ymin>{}</ymin>
            <ymax>{}</ymax>
        </bndbox>
        <truncated>0</truncated>
        <difficult>0</difficult>
    </object>
"""
# label转换
# label_dict = {
#     '不带电芯充电宝': 'coreless',
#     '带电芯充电宝': 'core',
# }
label_dict = {
    '不带电芯充电宝': 0,
    '带电芯充电宝': 1,
}


def convert_to_voc():
    """
    将充电宝数据集转换为voc格式
    :return:
    """
    categories = set()
    root = '/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/充电宝/遮挡问题/core'
    # 读取所有label
    for label_file_path in sorted(os.listdir(osp.join(root, 'Annotation'))):
        print(label_file_path)
        labels = pd.read_csv(osp.join(root, 'Annotation', label_file_path), delimiter=' ', header=None).values
        objects = ''
        for label in labels:
            # 只检测两类目标
            if label[1] not in label_dict.keys(): continue
            objects += '\n' + voc_object_template.format(
                label_dict[label[1]],
                label[2], label[4], label[3], label[5],
            )
            categories.add(label[1])
        # 　图片尺寸
        img_name = label_file_path[:-4] + '.jpg'
        img_size = Image.open(osp.join(root, 'Image', img_name)).size
        # 拼接label xml
        xml = voc_annotation_template.format(
            './portable',
            label_file_path[:-4] + '.jpg',
            img_size[0], img_size[1], 3,
            objects
        )
        with open('./portable/VOCdevkit/VOC2012/Annotations/{}.xml'.format(label_file_path[:-4]), 'w') as xml_file:
            xml_file.write(xml)


def split_dataset():
    # 读取所有数据集项目
    dataset = [name[:-4] for name in os.listdir('./portable/VOCdevkit/VOC2012/Annotations')]
    # 打乱顺序
    random.shuffle(dataset)
    # 划分
    train_split = dataset[:5500]
    val_split = dataset[5500:]

    pd.DataFrame(train_split).to_csv(
        './portable/VOCdevkit/VOC2012/ImageSets/Main/train.txt', header=False, index=False)
    pd.DataFrame(val_split).to_csv(
        './portable/VOCdevkit/VOC2012/ImageSets/Main/val.txt', header=False, index=False)
    print(len(train_split), len(val_split))


def get_portable_dataset(root, image_set, transforms, mode='instances'):
    """
    构建voc格式的充电包dataset
    :param root: 数据集根目录
    :param image_set: train/val
    :return:
    """

    return torchvision.datasets.VOCDetection(
        root=root,
        image_set=image_set,
        transforms=transforms  # 转换为tensor以及其他augmentation
    )


if __name__ == '__main__':
    convert_to_voc()
    # split_dataset()
    exit()
    dataset = get_portable_dataset(
        root='./portable',
        image_set='train',
        transforms=None
    )
    print(dataset)
    print(dataset.__getitem__(0))
