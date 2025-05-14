import os
import shutil

# 文件地址
source_dir = r'F:\MyProject\mmdetection\data\Autokary2022_1600x1600\test_labelme'
destination_dir = r'F:\MyProject\mmdetection\data\Autokary2022\test_new'

# 遍历文件地址a下的所有文件
for root, dirs, files in os.walk(source_dir):
    # 跳过文件地址a的最上级目录，直接处理下两级目录
    if root.count(os.sep) - source_dir.count(os.sep) == 1:
        # 复制每个文件到文件地址b
        for file in files:
            # 源文件路径
            file_path = os.path.join(root, file)
            # 目标文件路径
            dest_path = os.path.join(destination_dir, file)
            # 创建目标目录（如果不存在）
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            # 复制文件
            shutil.copy(file_path, dest_path)

print("文件转移完成！")
