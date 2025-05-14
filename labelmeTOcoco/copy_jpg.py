import os
import json
import shutil

def copy2dataset(file_src, annotation, file_dir):
    # 1. 读取 JSON
    with open(annotation, 'r', encoding='utf-8') as f:
        file_json = json.load(f)

    # 2. 如果目标目录不存在就创建
    os.makedirs(file_dir, exist_ok=True)

    # 3. 遍历源目录及所有子目录，建立一个“basename→真实路径”的映射（小写键）
    file_map = {}
    for root, _, files in os.walk(file_src):
        for fname in files:
            file_map[fname.lower()] = os.path.join(root, fname)

    # 4. 针对 JSON 中每条 image，取它的 basename（不含任何路径），然后查映射
    for img in file_json.get('images', []):
        raw_name = img.get('file_name', '')
        base_name = os.path.basename(raw_name).lower()

        if base_name in file_map:
            src_path = file_map[base_name]
            dst_path = os.path.join(file_dir, os.path.basename(src_path))
            try:
                shutil.copy2(src_path, dst_path)
                print(f"已复制: {base_name}")
            except Exception as e:
                print(f"复制失败 ({base_name}): {e}")
        else:
            print(f"警告: 在源目录及子目录中未找到文件 {raw_name}（basename={base_name}）")

if __name__ == '__main__':
    # src_dir = r'F:\MyProject\mmdetection\data\Autokary2022\train_new'
    # dst_dir = r'F:\MyProject\mmdetection\data\coco\train2017'
    # json_file = r'F:\MyProject\mmdetection\data\coco\train.json'

    # src_dir = r'F:\MyProject\mmdetection\data\Autokary2022\train_new'
    # dst_dir = r'F:\MyProject\mmdetection\data\coco\val2017'
    # json_file = r'F:\MyProject\mmdetection\data\coco\val.json'

    src_dir = r'F:\MyProject\mmdetection\data\Autokary2022\test_new'
    dst_dir = r'F:\MyProject\mmdetection\data\coco\test2017'
    json_file = r'F:\MyProject\mmdetection\data\coco\test.json'

    copy2dataset(src_dir, json_file, dst_dir)
