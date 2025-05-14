import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import random
import os


def visualize_coco_annotations(image_path, json_path, image_id=None, show_labels=True):
    """
    可视化COCO标注

    参数:
        image_path: 图片路径或目录
        json_path: COCO格式的json文件路径
        image_id: 要可视化的图片ID(如果为None，则使用image_path的文件名匹配)
        show_labels: 是否显示类别标签
    """
    # 加载COCO标注数据
    with open(json_path) as f:
        coco_data = json.load(f)

    # 创建类别ID到名称的映射
    categories = {c['id']: c['name'] for c in coco_data['categories']}

    # 创建类别ID到颜色的映射
    colors = {}
    for cat_id in categories.keys():
        colors[cat_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # 如果image_id未指定，尝试从文件名匹配
    if image_id is None:
        filename = os.path.basename(image_path)
        for img in coco_data['images']:
            if img['file_name'] == filename:
                image_id = img['id']
                break
        if image_id is None:
            raise ValueError(f"无法在JSON中找到图片: {filename}")

    # 获取图片信息
    image_info = None
    for img in coco_data['images']:
        if img['id'] == image_id:
            image_info = img
            break

    if image_info is None:
        raise ValueError(f"无法找到ID为 {image_id} 的图片")

    # 加载图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图片: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # 创建matplotlib图形
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)

    # 获取该图片的所有标注
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

    # 绘制每个标注
    patches = []
    colors_list = []
    for ann in annotations:
        cat_id = ann['category_id']
        color = colors[cat_id]

        # 处理多边形标注
        if 'segmentation' in ann and ann['segmentation']:
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((-1, 2))
                polygon = Polygon(poly, closed=True, fill=False,
                                  edgecolor=(color[0] / 255, color[1] / 255, color[2] / 255), linewidth=2)
                ax.add_patch(polygon)

                # 在第一个点上显示标签
                if show_labels:
                    label = categories[cat_id]
                    ax.text(poly[0, 0], poly[0, 1], label, color='white',
                            bbox=dict(facecolor=(color[0] / 255, color[1] / 255, color[2] / 255), alpha=0.7))

        # 处理bbox标注
        if 'bbox' in ann:
            bbox = ann['bbox']
            x, y, w, h = bbox
            rect = plt.Rectangle((x, y), w, h, linewidth=2,
                                 edgecolor=(color[0] / 255, color[1] / 255, color[2] / 255),
                                 facecolor='none')
            ax.add_patch(rect)

            # 在bbox左上角显示标签
            if show_labels:
                label = categories[cat_id]
                ax.text(x, y, label, color='white',
                        bbox=dict(facecolor=(color[0] / 255, color[1] / 255, color[2] / 255), alpha=0.7))

    plt.axis('off')
    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径
    json_file = "F:/MyProject/mmdetection/data/coco/annotations/instances_test2017.json"
    image_file = "F:/MyProject/mmdetection/data/coco/test2017/211025-003C_32_1_835_213_0.523.png"  # 替换为你想可视化的图片文件名

    # 可视化指定图片
    visualize_coco_annotations(image_file, json_file)