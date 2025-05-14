import os
import json
import random
import shutil
from pathlib import Path
from labelme import utils
from shapely.geometry import Polygon
from tqdm import tqdm

#gpt写的，是在htc需要一个stuffthing之后重新生成的，这里是用于train训练集，其中还取了15%作为验证集

# 设置路径
labelme_dir = 'F:/MyProject/mmdetection/data/Autokary2022/train_new'  # LabelMe JSON 和 PNG 文件的文件夹
output_dir = 'F:/MyProject/mmdetection/data/coco'  # 输出 COCO 格式的文件夹
image_dir = 'F:/MyProject/mmdetection/data/Autokary2022/train_new'  # 原始图像文件夹

# 标签定义（1到24）
labels = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13",
    "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24"
]

# COCO格式的数据结构
coco_data = {
    "images": [],
    "annotations": [],
    "categories": []
}

# 定义COCO类别
for i, label in enumerate(labels):
    coco_data["categories"].append({
        "id": i + 1,
        "name": label,
        "supercategory": "stuff"  # 可以根据需要调整为 "things" 或 "stuff"
    })

# 获取所有JSON文件
labelme_json_files = [f for f in os.listdir(labelme_dir) if f.endswith('.json')]

image_id = 0
annotation_id = 0

# 处理LabelMe的每个JSON文件
for json_file in tqdm(labelme_json_files):
    json_path = os.path.join(labelme_dir, json_file)

    # 读取LabelMe的JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        labelme_data = json.load(f)

    # 图像文件路径
    image_path = os.path.join(image_dir, labelme_data['imagePath'])
    image = utils.img_b64_to_arr(labelme_data['imageData'])  # 转换为数组
    height, width = image.shape[:2]

    # 创建COCO格式的图像信息
    image_id += 1
    coco_data["images"].append({
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": labelme_data['imagePath']
    })

    # 处理标注并转换为COCO格式
    for shape in labelme_data['shapes']:
        category_name = shape['label']
        if category_name in labels:
            category_id = labels.index(category_name) + 1  # COCO的类别ID从1开始

            # 获取多边形的坐标
            polygon = shape['points']
            segmentation = [polygon]  # COCO要求的是多边形的坐标列表

            # 计算多边形面积
            poly = Polygon(polygon)
            area = poly.area  # 使用shapely计算面积

            # 计算边界框
            minx, miny, maxx, maxy = poly.bounds
            bbox = [minx, miny, maxx - minx, maxy - miny]

            # 创建COCO标注信息
            annotation_id += 1
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })

# 将数据划分为训练集和验证集（15%为验证集）
total_images = coco_data['images']
num_val = int(len(total_images) * 0.15)
val_images = random.sample(total_images, num_val)
train_images = [img for img in total_images if img not in val_images]

# 创建训练集和验证集的COCO数据
train_coco = {
    "images": train_images,
    "annotations": [ann for ann in coco_data["annotations"] if ann["image_id"] in [img["id"] for img in train_images]],
    "categories": coco_data["categories"]
}

val_coco = {
    "images": val_images,
    "annotations": [ann for ann in coco_data["annotations"] if ann["image_id"] in [img["id"] for img in val_images]],
    "categories": coco_data["categories"]
}

# 保存COCO格式的JSON文件
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'train_coco.json'), 'w') as f:
    json.dump(train_coco, f, indent=4)

with open(os.path.join(output_dir, 'val_coco.json'), 'w') as f:
    json.dump(val_coco, f, indent=4)

# 将图像文件复制到新的文件夹（训练集和验证集的图像分别存储）
train_image_dir = os.path.join(output_dir, 'train2017')
val_image_dir = os.path.join(output_dir, 'val2017')

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)

# 将训练集和验证集的图像复制到相应的文件夹
for img in train_images:
    shutil.copy(os.path.join(image_dir, img['file_name']), os.path.join(train_image_dir, img['file_name']))

for img in val_images:
    shutil.copy(os.path.join(image_dir, img['file_name']), os.path.join(val_image_dir, img['file_name']))

print("数据转换并保存完成！")
