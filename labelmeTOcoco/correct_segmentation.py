import json

# 读取原标注文件
with open("F:/MyProject/mmdetection/data/coco/instances_train2017.json", "r") as f:
    data = json.load(f)

# 遍历所有标注，修正 segmentation
for ann in data["annotations"]:
    if "segmentation" in ann and isinstance(ann["segmentation"], list):
        new_seg = []
        for poly in ann["segmentation"]:
            # 如果 poly 是 [[x1,y1], [x2,y2], ...]，则展开成一维
            if isinstance(poly[0], list):
                flat_poly = [coord for point in poly for coord in point]
                new_seg.append(flat_poly)
            else:
                new_seg.append(poly)
        ann["segmentation"] = new_seg

# 保存修正后的文件
with open("F:/MyProject/mmdetection/data/coco/annotations/instances_train2017.json", "w") as f:
    json.dump(data, f, indent=2)