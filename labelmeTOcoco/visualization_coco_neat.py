import json
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from matplotlib import patches
from PIL import Image
import os

# 设置 COCO 数据集路径
coco_dir = 'F:/MyProject/mmdetection/data/coco'  # 你的 COCO 数据集路径
annotations_dir = os.path.join(coco_dir, 'annotations')  # COCO 标注文件夹

# 加载 COCO 数据集的标注文件
coco_file = os.path.join(annotations_dir, 'instances_train2017.json')

# 加载 COCO 数据集
coco = COCO(coco_file)

# 获取图像 ID（例如，随机选择一张图像）
image_id = coco.getImgIds()[0]  # 选择第一张图像

# 获取图像信息
image_info = coco.loadImgs(image_id)[0]

# 加载图像
image_path = os.path.join(coco_dir, 'train2017', image_info['file_name'])
image = Image.open(image_path)

# 获取该图像的所有注释
ann_ids = coco.getAnnIds(imgIds=image_id)
annotations = coco.loadAnns(ann_ids)

# 可视化图像和标注
plt.imshow(image)
ax = plt.gca()

# 绘制每个注释的边界框
for annotation in annotations:
    # 检查是否有边界框
    if 'bbox' in annotation:
        bbox = annotation['bbox']
        x, y, width, height = bbox
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # 如果是分割标注，绘制分割区域
    if 'segmentation' in annotation:
        for seg in annotation['segmentation']:
            # `seg` 是一个多边形坐标
            poly = list(zip(seg[::2], seg[1::2]))  # 提取多边形坐标
            poly = [(x, y) for x, y in poly]
            poly_patch = patches.Polygon(poly, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(poly_patch)

# 显示图像和标注
plt.axis('off')
plt.show()
