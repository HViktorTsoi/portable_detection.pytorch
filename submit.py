"""
Generate Submission and Calculate mAP
"""
import datetime
import time

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.faster_rcnn
from torchvision.models.detection.rpn import AnchorGenerator

from tools import utils
from data.dataset import get_portable_submit_dataset
from tools.voc_eval import voc_eval


def main(args):
    device = torch.device(args.device)

    # 数据集
    dataset_submit = get_portable_submit_dataset(args.image_root, args.imagesetfile)
    data_loader = torch.utils.data.DataLoader(
        dataset_submit, batch_size=1,
        sampler=torch.utils.data.SequentialSampler(dataset_submit),
        num_workers=args.workers, collate_fn=utils.collate_fn
    )

    # 模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        # 图像尺寸
        min_size=700, max_size=1333,
        # anchor大小
        rpn_anchor_generator=AnchorGenerator(
            sizes=((64,), (100,), (128,), (256,), (320,)),
            aspect_ratios=((0.8, 1.8),) * 5
        ),
        num_classes=args.num_classes + 1,
        pretrained=False
    )
    model.to(device)

    # 载入模型权重
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print('Loading Model From: ', args.model_path)

    print("Testing...")
    start_time = time.time()
    # 打开各类别的结果文件
    result_file = [open('./result/result_{}.txt'.format(_class), 'w') for _class in range(0, args.num_classes + 1)]

    model.eval()
    with torch.no_grad():
        for idx, (image, image_ids) in enumerate(data_loader):
            image = list(img.to(device) for img in image)

            torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(image)

            outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            print('|- {:04d} image:{} \ntime:{:.5f}  result label:{} result score{}  \n'.format(
                idx, image_ids[0], model_time, outputs[0]['labels'], outputs[0]['scores']))

            # 写入result文件
            boxes = outputs[0]['boxes'].int().cpu().numpy()
            for idx, box in enumerate(boxes):
                # 按类别写入
                _class = outputs[0]['labels'][idx].item()
                score = outputs[0]['scores'][idx].item()
                # 对score进行过滤
                if score < args.score_thresh: continue
                # 注意这里因为裁剪了白边 需要在结果的bb上加上部白边的偏移crop_y_min
                result_file[_class].write('{} {} {} {} {} {}\n'.format(
                    image_ids[0][:-4], score,
                    box[0], box[1] + dataset_submit.crop_y_min, box[2], box[3] + dataset_submit.crop_y_min,
                ))

            # import cv2
            # import matplotlib.pyplot as plt
            # img = image[0].cpu().numpy().transpose(1, 2, 0)
            # boxes = outputs[0]['boxes'].int().cpu().numpy()
            # img = cv2.resize(img, (1200, 700))
            # for box in boxes:
            #      cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=10)
            # cv2.imshow('', img)
            # cv2.waitKey(500000)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))

    # VOC测评 求各的ap
    do_voc_eval()


def do_voc_eval():
    """
    VOC测评 求各类别的ap

    :return:
    """
    print('\n\n\n')
    print('*' * 40)
    print('以VOC数据集标准计算mAP......')
    aps = []
    for _class in range(1, args.num_classes + 1):
        rec, prec, ap = voc_eval(
            detpath='./result/result_{}.txt',
            annopath=args.annopath,
            imagesetfile=args.imagesetfile,
            classname='{}'.format(_class),
            cachedir='./result/cache',
            ovthresh=args.iou_thresh,
        )
        aps.append(ap)
        print('|- class:{} AP:{:.3f}'.format(_class, ap))
    print('\nDone.\n')
    print('*' * 40)
    print('\n|- IOU thresh: {}'.format(args.iou_thresh))
    print('|- mAP: {}\n'.format(sum(aps) / args.num_classes))
    print('*' * 40)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--image-root', default='./data/portable/VOCdevkit/VOC2012/JPEGImages', help='图像文件夹路径')
    parser.add_argument('--annopath', default='./data/portable/VOCdevkit/VOC2012/Annotations/{}.xml',
                        help='标签路径, 形如./data/portable/VOCdevkit/VOC2012/Annotations/{}.xml')
    parser.add_argument('--imagesetfile', default='', help='测试集图像名称列表(注意仅为图像名称,不含路径和图像扩展名)')
    parser.add_argument('--iou-thresh', default=0.5, help='IOU阈值', type=float)
    parser.add_argument('--score-thresh', default=0.0, help='bbox置信度阈值,小于此阈值的结果将被过滤掉', type=float)

    parser.add_argument('--num-classes', default=2, type=int, help='print frequency')
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--model-path', default='', help='resume from checkpoint')

    args = parser.parse_args()
    print(args)

    main(args)
