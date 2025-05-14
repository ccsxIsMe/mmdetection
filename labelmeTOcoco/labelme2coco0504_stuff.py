import os
import json
import numpy as np
from PIL import Image
from skimage.draw import polygon
from tqdm import tqdm

# 设置路径
coco_dir = 'F:/MyProject/mmdetection/data/coco'  # COCO 数据集路径
annotations_dir = os.path.join(coco_dir, 'annotations')  # COCO 标注文件夹
stuffthingmaps_dir = os.path.join(coco_dir, 'stuffthingmaps')  # COCO 背景分割图文件夹

# 创建stuffthingmaps目录（分别为训练集、验证集、测试集创建子文件夹）
os.makedirs(os.path.join(stuffthingmaps_dir, 'train2017'), exist_ok=True)
os.makedirs(os.path.join(stuffthingmaps_dir, 'val2017'), exist_ok=True)
os.makedirs(os.path.join(stuffthingmaps_dir, 'test2017'), exist_ok=True)

# 加载现有的COCO数据集（train_coco.json, val_coco.json, test_coco.json）
def load_coco_data(coco_file):
    with open(coco_file, 'r') as f:
        return json.load(f)

# 获取 COCO 标注数据文件
train_coco_file = os.path.join(annotations_dir, 'instances_train2017.json')
val_coco_file = os.path.join(annotations_dir, 'instances_val2017.json')
test_coco_file = os.path.join(annotations_dir, 'instances_test2017.json')

# 加载 COCO 数据集
train_coco = load_coco_data(train_coco_file)
val_coco = load_coco_data(val_coco_file)
test_coco = load_coco_data(test_coco_file)

# 获取所有annotations
all_annotations = train_coco['annotations'] + val_coco['annotations'] + test_coco['annotations']

# 为每个图像生成stuffthingmaps
def generate_stuffthingmaps(coco_data, annotations, stuffthingmaps_dir):
    # 创建图像文件名字典
    image_names = {img['file_name']: img for img in coco_data['images']}

    for image_info in tqdm(coco_data['images']):
        # 创建空白分割图
        img_width = image_info['width']
        img_height = image_info['height']
        segmentation_map = np.zeros((img_height, img_width), dtype=np.uint8)

        # 获取该图像的所有annotations
        image_annotations = [ann for ann in annotations if ann['image_id'] == image_info['id']]

        for annotation in image_annotations:
            category_id = annotation['category_id']
            if category_id == 0:  # 0通常代表背景区域
                continue
            segmentation = annotation['segmentation']
            for poly in segmentation:
                # 转换为像素坐标并绘制在分割图上
                poly_points = np.array(poly).reshape((-1, 2))
                rr, cc = polygon(poly_points[:, 1], poly_points[:, 0], (img_height, img_width))
                segmentation_map[rr, cc] = category_id

        # 根据所属数据集将图像保存到对应文件夹
        output_img_path = None
        if image_info['file_name'] in image_names:
            if image_info['file_name'] in [img['file_name'] for img in train_coco['images']]:
                output_img_path = os.path.join(stuffthingmaps_dir, 'train2017', f"{image_info['file_name']}")
            elif image_info['file_name'] in [img['file_name'] for img in val_coco['images']]:
                output_img_path = os.path.join(stuffthingmaps_dir, 'val2017', f"{image_info['file_name']}")
            elif image_info['file_name'] in [img['file_name'] for img in test_coco['images']]:
                output_img_path = os.path.join(stuffthingmaps_dir, 'test2017', f"{image_info['file_name']}")

        # 如果找不到对应路径，跳过该图像
        if output_img_path:
            segmentation_image = Image.fromarray(segmentation_map)
            segmentation_image.save(output_img_path)
        else:
            print(f"跳过了图像 {image_info['file_name']}，未找到对应的分类")


# 调用生成函数
generate_stuffthingmaps(train_coco, all_annotations, stuffthingmaps_dir)
generate_stuffthingmaps(val_coco, all_annotations, stuffthingmaps_dir)
generate_stuffthingmaps(test_coco, all_annotations, stuffthingmaps_dir)

print("Stuffthingmaps已生成并保存！")

