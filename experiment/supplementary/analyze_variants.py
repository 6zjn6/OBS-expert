#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

"""
分析异体字数据
"""

def analyze_variants():
    """分析seen和unseen数据集中的异体字"""
    
    # 读取seen数据集
    seen_chars = set()
    seen_data = {}
    base_data_dir = Path(__file__).resolve().parents[1] / 'data'
    with open(str(base_data_dir / 'character_explanations_CN_seen.csv'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过标题行
            if line.strip():
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    char = parts[0]
                    explanation = parts[1]
                    seen_chars.add(char)
                    seen_data[char] = explanation
    
    # 读取unseen数据集
    unseen_chars = set()
    unseen_data = {}
    with open(str(base_data_dir / 'character_explanations_CN_unseen.csv'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过标题行
            if line.strip():
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    char = parts[0]
                    explanation = parts[1]
                    unseen_chars.add(char)
                    unseen_data[char] = explanation
    
    # 找出重复字符（异体字）
    variant_chars = seen_chars.intersection(unseen_chars)
    
    print(f"Seen数据集字符数量: {len(seen_chars)}")
    print(f"Unseen数据集字符数量: {len(unseen_chars)}")
    print(f"异体字数量: {len(variant_chars)}")
    print(f"\n异体字列表:")
    
    for char in sorted(variant_chars):
        print(f"\n字符: {char}")
        print(f"Seen数据集解释: {seen_data[char]}")
        print(f"Unseen数据集解释: {unseen_data[char]}")
        print("-" * 50)
    
    return variant_chars, seen_data, unseen_data

if __name__ == "__main__":
    analyze_variants()
