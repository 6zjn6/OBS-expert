#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - 使用PrototypeClassifier和现有GPT API
"""

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import pandas as pd
import re
import os
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from PrototypeClassifier import get_model, get_prototype_classifier
from chatgpt import chat_with_gpt_new_noimage, chat_with_gpt_new_bothimage

# 统一项目根目录（便于从子目录运行时也能找到 data/**）
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# 全局变量
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_prototype_model():
    """
    获取PrototypeClassifier模型
    
    返回:
        model: 预训练的DinoV2模型
        class_prototypes: 类别原型字典
        train_classes: 训练类别列表
        mean: 标准化均值
        std: 标准化标准差
    """
    print("🔧 加载PrototypeClassifier模型...")
    
    try:
        # 在调用内部相对路径前，确保工作目录为项目根目录
        prev_cwd = os.getcwd()
        os.chdir(ROOT_DIR)

        # 关键数据目录存在性检查（相对项目根目录）
        org_dir = os.path.join(ROOT_DIR, 'data', 'organized_radicals')
        if not os.path.exists(org_dir):
            print(f"❌ 训练数据路径不存在: {os.path.relpath(org_dir, ROOT_DIR)}")

        # 使用现有的PrototypeClassifier（相对路径将在 ROOT_DIR 下解析）
        model, class_prototypes, train_classes = get_prototype_classifier()
        
        # 计算标准化参数（这里需要根据实际训练数据计算）
        # 暂时使用默认值，实际使用时需要从训练数据中计算
        mean = np.zeros(768)  # DinoV2特征维度
        std = np.ones(768)
        
        print(f"✅ 成功加载模型，类别数: {len(train_classes)}")
        return model, class_prototypes, train_classes, std, mean
        
    except Exception as e:
        print(f"❌ 加载PrototypeClassifier失败: {e}")
        print("⚠️  请确保 data/organized_radicals 目录存在（相对项目根目录）")
        return None, None, None, None, None
    finally:
        # 恢复原工作目录，避免影响其他逻辑
        try:
            os.chdir(prev_cwd)
        except Exception:
            pass

def get_possible_radical_prototype(radical_image, class_prototypes, train_classes, model, all_radical_list, std, mean, topk=5):
    """
    使用PrototypeClassifier预测部首（不进行过滤）
    
    参数:
        radical_image: 部首图像
        class_prototypes: 类别原型字典
        train_classes: 训练类别列表
        model: 预训练模型
        all_radical_list: 所有可能的部首列表（仅用于参考）
        std: 标准化标准差
        mean: 标准化均值
        topk: 返回前k个结果
        
    返回:
        list: 前k个可能的部首（基于相似度排序）
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 预处理图像
    image = transform(radical_image)
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 提取特征
        feature = model(image).cpu().numpy()
        
    # 标准化特征
    feature = (feature - mean) / std
    
    # 计算与所有原型的余弦相似度
    similarities = {}
    for class_idx, prototype in class_prototypes.items():
        if class_idx < len(train_classes):
            class_name = train_classes[class_idx]
            # 计算余弦相似度
            sim = cosine_similarity(feature.reshape(1, -1), prototype.reshape(1, -1))[0][0]
            similarities[class_name] = sim
    
    # 按相似度排序
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # 直接返回相似度最高的topk个部首，不进行过滤
    top_results = []
    for class_name, similarity in sorted_similarities[:topk]:
        top_results.append(class_name)
    
    print(f"预测的部首相似度: {sorted_similarities[:topk]}")
    print(f"预测的部首: {top_results}")
    
    return top_results if top_results else None

def get_separation(explanation):
    """
    从字符解释中提取部首信息
    
    参数:
        explanation: 字符的解释文本
        
    返回:
        count: 部首数量
        radical_list: 部首列表
        ground_truth: 原始解释文本
    """
    # 这里需要根据您的实际数据格式实现
    # 示例实现：从解释文本中提取部首信息
    
    radical_list = []
    ground_truth = explanation
    
    # 使用正则表达式提取部首
    # 根据您的数据格式调整正则表达式
    radical_pattern = r'[''""]([^''""]+)[''""]'
    matches = re.findall(radical_pattern, explanation)
    
    for match in matches:
        if match not in radical_list:
            radical_list.append(match)
    
    # 如果没有找到部首，尝试其他方法
    if not radical_list:
        # 可以尝试从radical_explanation.csv中查找
        try:
            df_rad = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'radical_explanation.csv'))
            # 查找包含该字符的部首
            related_radicals = df_rad[df_rad['Part_of_Character'] == explanation.split('，')[0]]['Radical'].unique()
            radical_list = list(related_radicals)
        except pd.errors.ParserError as e:
            print(f"⚠️ 部首CSV解析错误: {e}")
            try:
                # 尝试使用更宽松的参数
                df_rad = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'radical_explanation.csv'), 
                                   on_bad_lines='skip',
                                   encoding='utf-8-sig',
                                   quoting=csv.QUOTE_NONE,
                                   error_bad_lines=False)
                related_radicals = df_rad[df_rad['Part_of_Character'] == explanation.split('，')[0]]['Radical'].unique()
                radical_list = list(related_radicals)
            except:
                pass
        except:
            pass
    
    return len(radical_list), radical_list, ground_truth

def chat_with_gpt_new_noimage_wrapper(image_path, possible_radicals):
    """
    无图片的GPT对话函数包装器
    
    参数:
        image_path: 图片路径
        possible_radicals: 可能的部首列表
        
    返回:
        str: GPT的回复文本
    """
    try:
        # 直接调用现有的GPT函数
        response = chat_with_gpt_new_noimage(image_path, possible_radicals)
        return response
    except Exception as e:
        print(f"❌ GPT API调用失败: {e}")
        return f"The oracle bone character is composed of {possible_radicals}"

def chat_with_gpt_new_bothimage_wrapper(image_path, radical_image_paths, possible_radicals):
    """
    带图片的GPT对话函数包装器
    
    参数:
        image_path: 原图路径
        radical_image_paths: 部首图片路径列表
        possible_radicals: 可能的部首列表
        
    返回:
        str: GPT的回复文本
    """
    try:
        # 直接调用现有的GPT函数
        response = chat_with_gpt_new_bothimage(image_path, radical_image_paths, possible_radicals)
        return response
    except Exception as e:
        print(f"❌ GPT API调用失败: {e}")
        return f"The oracle bone character is composed of {possible_radicals}"

def chat_with_gpt_new_noimage_english_wrapper(image_path, possible_radicals):
    """Wrapper for English output format - no image mode"""
    from chatgpt import chat_with_gpt_new_noimage_english
    return chat_with_gpt_new_noimage_english(image_path, possible_radicals)

def chat_with_gpt_new_bothimage_english_wrapper(image_path, radical_image_paths, possible_radicals):
    """Wrapper for English output format - both image mode"""
    from chatgpt import chat_with_gpt_new_bothimage_english
    return chat_with_gpt_new_bothimage_english(image_path, radical_image_paths, possible_radicals)

# 辅助函数
def load_radical_images_from_organized():
    """
    从organized_radicals文件夹加载部首图片用于训练PrototypeClassifier
    """
    # 这里需要实现从data/organized_radicals/加载训练数据的逻辑
    # 用于训练PrototypeClassifier
    pass

def extract_features_from_image(model, image_path):
    """
    使用预训练模型提取图片特征
    """
    # 实现特征提取逻辑
    pass

def prepare_training_data():
    """
    准备训练数据用于PrototypeClassifier
    """
    # 从data/organized_radicals/准备训练数据
    # 创建ImageFolder格式的数据集
    pass 