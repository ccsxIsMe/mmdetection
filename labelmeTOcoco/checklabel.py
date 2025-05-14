#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys

def get_labelme_labels(json_path):
    """
    读取 LabelMe 导出的 JSON 文件，返回其中所有的 label 名称集合。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # shapes 字段中每个元素的 'label' 即为对象类别
    labels = {shape.get('label') for shape in data.get('shapes', [])}
    return labels

def main():
    if len(sys.argv) != 2:
        print(f"用法: {sys.argv[0]} path/to/labelme.json")
        sys.exit(1)

    json_file = sys.argv[1]
    try:
        labels = get_labelme_labels(json_file)
    except Exception as e:
        print(f"读取或解析 JSON 时出错：{e}")
        sys.exit(1)

    if labels:
        print("在文件中找到以下 labels：")
        for lbl in sorted(labels):
            print(f" - {lbl}")
    else:
        print("未在 'shapes' 中找到任何 label。")

if __name__ == '__main__':
    main()
