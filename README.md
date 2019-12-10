## 数据集
需要先把训练集处理成VOC格式,放在data目录下,以数据集名称为"portable"为例,目录结构为
```markdown
data
|-portable
|--VOCdevkit
|---VOC2012
|----Annotations
|-----core_battery00000003.xml
|-----core_battery00000004.xml
|-----....
|----ImageSets
|-----Main
|------train.txt(里边是不包含图片名的文件列表)
|------test.txt
|------val.txt
|----JPEGImages
|-----core_battery00000003.jpg
|-----core_battery00000004.jpg
|-----...
```

## 训练(支持分布式 可以单机多显卡或者多机)
python -m torch.distributed.launch --nproc_per_node=3
--use_env train.py --dataset portable --data-path ./data/portable -j 8 --print-freq 20 --model fasterrcnn_resnet50_fpn \
--batch-size 4 --epoch 25 --lr-steps 0 5 15 --lr 0.003 --lr-gamma 0.7 --aspect-ratio-group-factor 3 --output-dir checkpoints \
--resume ./checkpoints/model_5.pth

## 测试(计算在测试集上的mAP)
python submit.py --image-root data/portable/VOCdevkit/VOC2012/JPEGImages \
--imagesetfile /home/hviktortsoi/Code/portable_detection/data/portable/VOCdevkit/VOC2012/ImageSets/Main/test.txt
--annopath ./data/portable/VOCdevkit/VOC2012/Annotations/{}.xml
--iou-thresh 0.5 --model-path checkpoints/model_submit.pth \
