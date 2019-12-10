import random
import torch

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class CropWhiteBorder(object):
    def __init__(self, crop_y_min=300, crop_y_max=1000):
        self.crop_y_min, self.crop_y_max = crop_y_min, crop_y_max

    def __call__(self, image, target):
        width, height = image.size
        # 裁剪图像白边
        image = image.crop((0, self.crop_y_min, width, self.crop_y_max))
        for idx in range(len(target['boxes'])):
            target['boxes'][idx][1] -= self.crop_y_min
            target['boxes'][idx][3] -= self.crop_y_min
        return image, target


class VocTargetToTorchVision(object):

    def _parse_object(self, obj):
        bndbox = obj['bndbox']
        box = [int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])]
        label = int(obj['name'])
        area = (box[3] - box[1]) * (box[2] - box[0])
        return box, label, area

    def __call__(self, image, target):
        objects = target['annotation']['object']
        boxes = []
        labels = []
        areas = []
        # 处理单个obj的情况
        if isinstance(objects, dict):
            objects = [objects]
        for obj in objects:
            box, label, area = self._parse_object(obj)
            boxes.append(box)
            labels.append(label)
            areas.append(area)
        # 拼接转换后的label
        cvt_target = {
            "boxes": boxes,
            "labels": labels,
            "area": areas,
            "iscrowd": len(objects),
            "image_id": [hash(target['annotation']['filename'])],
        }
        return image, cvt_target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = {
            "boxes": torch.as_tensor(target['boxes'], dtype=torch.float32),
            "labels": torch.as_tensor(target['labels'], dtype=torch.int64),
            "area": torch.as_tensor(target['area'], dtype=torch.float32),
            "iscrowd": torch.zeros(target['iscrowd'], dtype=torch.int64),
            "image_id": torch.as_tensor(target['image_id'], dtype=torch.int64)
        }
        return image, target
