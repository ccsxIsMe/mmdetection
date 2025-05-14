import os
import shutil

this_dir_path = 'F:/MyProject/mmdetection/data/Autokary2022_1600x1600/test_labelme/'
destination_directory = 'F:/MyProject/mmdetection/data/Autokary2022/test_new/'
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

subdirectories = []  # 存储子目录名称的列表
# 遍历目录
for root, dirs, files in os.walk(this_dir_path):
    for dir_name in dirs:
        subdirectories.append(dir_name)

# for directory in os.listdir(this_dir_path):
for directory in subdirectories:
    for file in os.listdir(os.path.join(this_dir_path, directory)):
        source_file = os.path.join(os.path.join(this_dir_path, directory), file)
        if directory == '211029-009C':
            # 这四个文件采用的是"line strip"无法转化为coco
            if os.path.splitext(file)[0] == '107_1_590_345_0.513' or \
                    os.path.splitext(file)[0] == '129_3_688_378_0.848' or \
                    os.path.splitext(file)[0] == '10_1_737_180_0.571' or \
                    os.path.splitext(file)[0] == '127_2_590_378_0.492':
                continue
        elif os.path.splitext(source_file)[-1] == '.png':
            new_file_path = destination_directory + file
            print(source_file + '---->' + new_file_path)
            destination_file = os.path.join(destination_directory, file)
            shutil.copy2(source_file, destination_file)
        elif os.path.splitext(source_file)[-1] == '.json':
            new_file_path = destination_directory + file
            print(source_file + '---->' + new_file_path)
            destination_file = os.path.join(destination_directory, file)
            shutil.copy2(source_file, destination_file)