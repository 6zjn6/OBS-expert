#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import csv
import os

def robust_read_csv(file_path, expected_columns=None):
    """
    Args:
        file_path: CSV文件路径
        expected_columns: 期望的列数，如果为None则自动检测
    
    Returns:
        pandas.DataFrame: 读取的数据框
    """
    print(f"📖 读取CSV文件: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 方法1: 尝试直接读取
    try:
        df = pd.read_csv(file_path)
        print("✅ 直接读取成功")
        return df
    except Exception as e:
        print(f"⚠️ 直接读取失败: {e}")
    
    # 方法2: 使用错误处理参数
    try:
        df = pd.read_csv(file_path, 
                        on_bad_lines='skip',
                        encoding='utf-8-sig',
                        quoting=csv.QUOTE_NONE,
                        error_bad_lines=False)
        print("✅ 错误处理读取成功")
        return df
    except Exception as e:
        print(f"⚠️ 错误处理读取失败: {e}")
    
    # 方法3: 手动读取并清理
    print("🔧 使用手动读取方法...")
    rows = []
    
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        if expected_columns is None:
            expected_columns = len(header)
        
        print(f"📋 标题行: {header} (期望字段数: {expected_columns})")
        rows.append(header)
        
        for i, row in enumerate(reader, 1):
            # 清理每一行
            cleaned_row = []
            for field in row:
                # 移除可能的隐藏字符和多余空格
                cleaned_field = field.strip().replace('\ufeff', '').replace('\u200b', '')
                cleaned_row.append(cleaned_field)
            
            # 处理字段数问题
            if len(cleaned_row) == expected_columns:
                # 字段数正确
                rows.append(cleaned_row)
            elif len(cleaned_row) > expected_columns:
                # 字段数过多，合并多余的字段到最后一个字段
                print(f"🔧 修复行 {i+1}: 字段数过多 ({len(cleaned_row)} -> {expected_columns})")
                merged_row = cleaned_row[:expected_columns-1] + [','.join(cleaned_row[expected_columns-1:])]
                rows.append(merged_row)
            elif len(cleaned_row) < expected_columns:
                # 字段数不足，跳过
                print(f"⚠️ 跳过行 {i+1}: 字段数不足 ({len(cleaned_row)} < {expected_columns})")
            else:
                rows.append(cleaned_row)
    
    # 创建DataFrame
    df = pd.DataFrame(rows[1:], columns=rows[0])
    print(f"✅ 手动读取成功: {df.shape}")
    
    return df

# 测试函数
def test_robust_reader():
    """测试强大的CSV读取器"""
    csv_file = 'data/radical_explanation.csv'
    
    try:
        df = robust_read_csv(csv_file, expected_columns=4)
        print(f"🎉 测试成功: {df.shape}")
        print(f"   列名: {df.columns.tolist()}")
        print(f"   部首数量: {df['Radical'].nunique()}")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    test_robust_reader()
