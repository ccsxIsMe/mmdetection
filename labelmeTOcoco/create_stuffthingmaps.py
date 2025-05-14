import os
import json
import numpy as np
from PIL import Image
from pycocotools.coco import COCO


def generate_stuffthingmaps(annotations_file, image_dir, output_dir):
    # 加载 COCO 数据集的 annotations 文件
    coco = COCO(annotations_file)

    # 获取所有的图像ID
    image_ids = coco.getImgIds()

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取每个图像的分割标注并生成掩码图
    for image_id in image_ids:
        # 获取图像信息
        img_info = coco.loadImgs(image_id)[0]
        img_file = img_info['file_name']

        # 获取图像的大小
        width, height = img_info['width'], img_info['height']

        # 创建一个空的掩码图（全为0）
        mask = np.zeros((height, width), dtype=np.uint8)

        # 获取该图像的所有标注
        ann_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(ann_ids)

        # 遍历每个标注并填充掩码图
        for ann in annotations:
            print(f"Annotations for image {img_file}: {ann}")
            if 'segmentation' in ann:
                # 处理多边形分割
                if isinstance(ann['segmentation'], list):
                    for poly in ann['segmentation']:
                        # 将每个多边形绘制在掩码图上
                        mask = coco.annToMask(ann)
                else:
                    # 处理单个多边形
                    mask = coco.annToMask(ann)

            # 使用类别ID作为掩码的值
            mask[mask == 1] = ann['category_id']

        # 将掩码图保存为 PNG 文件
        print(f"Saving mask for image {img_file}, shape: {mask.shape}")
        output_path = os.path.join(output_dir, img_file.replace('.jpg', '.png'))
        Image.fromarray(mask).save(output_path)


# 定义输入和输出路径
annotations_file = 'F:/MyProject/mmdetection/data/coco/annotations/instances_train2017.json'  # 训练集注释文件
image_dir = 'F:/MyProject/mmdetection/data/coco/train2017/'  # 图像文件夹路径
output_dir = 'F:/MyProject/mmdetection/data/coco/stuffthingmaps/train2017/'  # 输出的 stuffthingmaps 目录

generate_stuffthingmaps(annotations_file, image_dir, output_dir)
