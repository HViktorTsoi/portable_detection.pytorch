import os
import random
import sys
import transforms as T
import torch.utils.data as data
from torchvision.transforms import functional as F

import os.path as osp
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from sklearn.cluster.k_means_ import KMeans

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
    '带电芯充电宝': 1,
    '不带电芯充电宝': 2,
}


def calc_anchors():
    """
    计算anchors
    :return:
    """
    ratios = []
    sizes = []
    bbx = []
    root = '/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/充电宝/遮挡问题/core'
    for label_file_path in sorted(os.listdir(osp.join(root, 'Annotation'))):
        labels = pd.read_csv(osp.join(root, 'Annotation', label_file_path), delimiter=' ', header=None).values
        for label in labels:
            _, name, xmin, ymin, xmax, ymax = label
            if name not in label_dict.keys() or ymax - ymin < 2:
                # print(label_file_path, label)
                continue

            ratios.append((xmax - xmin) / (ymax - ymin))
            sizes.append([xmax - xmin, ymax - ymin])
            bbx.append([xmin, ymin, xmax, ymax])
    # 聚类分析anchor
    cluster = KMeans(n_clusters=3).fit(sizes)
    # 两种分辨率
    print(cluster.cluster_centers_ * (1333 / 2000))
    cluster = KMeans(n_clusters=2).fit(np.array(ratios).reshape(-1, 1))
    print(cluster.cluster_centers_)

    plt.hist(ratios, bins=10)
    plt.show()
    plt.hist2d(np.array(sizes)[:, 0], np.array(sizes)[:, 1], bins=20)
    plt.show()
    # bbx = np.array(bbx)
    # plt.hist(bbx[:, 1])
    # plt.show()
    # print(np.max(bbx, axis=0))
    # print(np.min(bbx, axis=0))

    # # 查看标定框分布
    # centers_x = labels[:, 1] + (labels[:, 5] - labels[:, 1]) / 2
    # centers_y = labels[:, 2] + (labels[:, 6] - labels[:, 2]) / 2
    # dist, edges = np.histogram(centers_x, bins=50)
    # plt.subplot(2, 1, 1)
    # # print(edges, dist)
    # plt.scatter(edges[:-1], dist)
    # plt.xlim(0, 1200)
    #
    # plt.subplot(2, 1, 2)
    # dist, edges = np.histogram(centers_y, bins=50)
    # # print(edges, dist)
    # plt.scatter(edges[:-1], dist)
    # plt.xlim(0, 1000)
    # plt.show()


def convert_to_voc():
    """
    将充电宝数据集转换为voc格式
    :return:
    """
    categories = set()
    root = '/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/充电宝/遮挡问题/core'
    img_sizes = []
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


def crop_images():
    root = '/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/充电宝/遮挡问题/core'
    img_sizes = []
    names = set()
    # 读取所有label
    for img_name in sorted(os.listdir(osp.join(root, 'Image'))):
        names.add(hash(img_name))
        print(len(names))
        print(img_name)
        img = Image.open(osp.join(root, 'Image', img_name))
        # plt.imshow(np.array(img)[300:850, ...])
        # plt.show()
        # exit()
        img_size = img.size
        img_sizes.append(list(img_size))
    img_sizes = np.array(img_sizes)
    plt.hist(img_sizes[:, 0], bins=20)
    plt.show()
    plt.hist(img_sizes[:, 1], bins=20)
    plt.show()


class PortableDataset(torchvision.datasets.VOCDetection):
    """
    充电宝数据集
    """

    def __init__(self, root, image_set, transforms):
        super().__init__(root, image_set=image_set, transforms=transforms)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target


class PortableSubmitDataset(data.Dataset):
    """
    充电宝submit数据集
    """

    def __init__(self, root, imagesetfile, crop_y_min=300, crop_y_max=1000):
        with open(imagesetfile) as file:
            self.imageset = [osp.join(root, line.replace('\n', '') + '.jpg') for line in file.readlines()]
        self.crop_y_min, self.crop_y_max = crop_y_min, crop_y_max
        print('Submission set size: ', len(self.imageset))

    def __getitem__(self, index):
        # 返回图像和图像id
        # 因为是测试集 不需要返回target
        image = Image.open(self.imageset[index])
        image_id = os.path.split(self.imageset[index])[-1]

        # 裁剪图像白边
        width, height = image.size
        image = image.crop((0, self.crop_y_min, width, self.crop_y_max))

        # 转换为tensor等
        image = F.to_tensor(image)
        return image, image_id

    def __len__(self):
        return len(self.imageset)


def get_portable_dataset(root, image_set, transforms, mode='instances'):
    """
    构建voc格式的充电宝dataset
    :param root: 数据集根目录
    :param image_set: train/val
    :return:
    """
    voc_dataset = PortableDataset(
        root=root,
        image_set=image_set,
        transforms=transforms  # 转换为tensor以及其他augmentation
    )
    return voc_dataset


def get_portable_submit_dataset(root, imagesetfile):
    return PortableSubmitDataset(root, imagesetfile)


if __name__ == '__main__':
    # convert_to_voc()
    # split_dataset()
    # calc_anchors()
    # crop_images()
    # exit()
    transforms = T.Compose([
        T.VocTargetToTorchVision(),
        T.CropWhiteBorder(),
        T.ToTensor(),
    ])
    dataset = get_portable_dataset(
        root='./portable',
        image_set='train',
        transforms=transforms
    )
    print(dataset)
    print(dataset.__getitem__(0)[0].size())
