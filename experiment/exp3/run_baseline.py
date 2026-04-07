# from KG_construct import KG_construct_new  # Baseline不需要KG构建
import itertools
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import numpy as np
from torchvision import transforms
import torch
import csv
from sklearn.model_selection import train_test_split
import re
from config import get_prototype_model, get_possible_radical_prototype, get_separation
from chatgpt_rag import (
    chat_with_gpt_variant_explanation
    # warm_up_cache,  # Baseline不需要缓存预热
    # get_cache_stats,  # Baseline不需要缓存统计
    # search_exact_character,  # Baseline不使用数据库搜索
    # search_character_by_radical,
    # search_radical_explanation
)
import argparse
import os
import random
from pathlib import Path

# 全局变量
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def split_data_for_kg_and_test(csv_file, train_ratio=0.7, random_seed=42):
    """
    将数据分割为训练集（用于构建KG）和测试集（用于测试LLM）
    
    Args:
        csv_file: CSV文件路径
        train_ratio: 训练集比例，默认0.7（7/10）
        random_seed: 随机种子，确保可重复性
    
    Returns:
        train_df: 训练集数据框
        test_df: 测试集数据框
    """
    print(f"📊 开始数据分割...")
    print(f"   训练集比例: {train_ratio:.1%}")
    print(f"   测试集比例: {1-train_ratio:.1%}")
    print(f"   随机种子: {random_seed}")
    
    # 读取数据
    df = pd.read_csv(csv_file)
    print(f"   总数据量: {len(df)} 个字符")
    
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 随机打乱数据
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # 计算分割点
    split_point = int(len(df_shuffled) * train_ratio)
    
    # 分割数据
    train_df = df_shuffled.iloc[:split_point].copy()
    test_df = df_shuffled.iloc[split_point:].copy()
    
    print(f"   训练集数量: {len(train_df)} 个字符")
    print(f"   测试集数量: {len(test_df)} 个字符")
    
    # 保存分割后的数据
    train_file = csv_file.replace('.csv', '_seen.csv')
    test_file = csv_file.replace('.csv', '_unseen.csv')
    
    train_df.to_csv(train_file, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_file, index=False, encoding='utf-8-sig')
    
    print(f"   训练集已保存到: {train_file}")
    print(f"   测试集已保存到: {test_file}")
    
    return train_df, test_df

def get_radical_images_from_img_zi(character, image_name):
    """从img_zi文件夹获取部首图像"""
    img_zi_dir = str((Path(__file__).resolve().parents[1] / 'data' / 'img_zi' / character))
    radical_images = []
    radical_image_paths = []
    target_radical = []
    
    if not os.path.exists(img_zi_dir):
        print(f"目录不存在: {img_zi_dir}")
        return [], [], []
    
    # 查找对应的图片文件夹
    for file in os.listdir(img_zi_dir):
        if file.endswith('.jpg') and image_name in file:
            base_name = file[:-4]  # 去掉.jpg后缀
            print(f"找到图片: {file}")
            
            # 查找对应的部首图片
            for radical_file in os.listdir(img_zi_dir):
                if radical_file.endswith('.png') and base_name in radical_file and '_' in radical_file:
                    # 提取部首名称
                    radical_name = radical_file.split('_')[-1][:-4]  # 去掉.png后缀
                    target_radical.append(radical_name)
                    
                    # 读取部首图片
                    image_path = os.path.join(img_zi_dir, radical_file)
                    image = Image.open(image_path)
                    radical_image_paths.append(image_path)
                    
                    # 转换为RGB格式
                    image = np.array(image)
                    if len(image.shape) == 2:
                        image = np.expand_dims(image, axis=-1)
                    if image.shape[-1] == 1:
                        image = np.repeat(image, 3, axis=-1)
                    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    radical_images.append(image)
            
            break
    
    return radical_images, radical_image_paths, target_radical


def clean_llm_output(output_text, character=None, is_baseline=False):
    """清理LLM输出，只保留释义内容，如果为空则返回空字符串让LLM自己处理"""
    
    if not output_text or output_text.strip() == "":
        print(f"    ⚠️ LLM输出为空，返回空字符串")
        return ""
    
    # 直接清理输出，因为现在LLM直接输出释义
    cleaned_output = output_text.strip()
    
    # 移除可能的格式标记（但保留中文内容和标点符号）
    # 只移除开头的纯格式标记，保留内容
    cleaned_output = re.sub(r'^[-\*•\s]+', '', cleaned_output)
    
    # 只移除结尾的纯格式标记，保留内容
    cleaned_output = re.sub(r'[-\*•\s]+$', '', cleaned_output)
    
    # 检查清理后的输出是否为空
    if not cleaned_output:
        print(f"    ⚠️ 清理后输出为空，返回空字符串")
        return ""
    
    # 如果输出太长，只保留前300个字符（释义通常需要更多文字）
    if len(cleaned_output) > 300:
        cleaned_output = cleaned_output[:300] + "..."
    
    return cleaned_output

def process_test_characters_baseline_only(test_df, force_restart=False):
    """处理测试集字符，仅使用baseline pipeline生成甲骨字释义，支持断点续传"""
    
    # 使用测试集数据
    All_zi = test_df
    print(f"成功读取测试集数据: {len(All_zi)} 个字符")
    
    # 创建输出目录 - 修改为baseoutput
    output_dir = 'baseoutput'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取大模型型号用于文件名，与chatgpt_rag.py中的get_llm()保持一致
    # baseline的CSV文件路径
    baseline_file = f'{output_dir}/test_set_baseline.csv'
    
    # 检查是否强制重新开始
    if force_restart:
        if os.path.exists(baseline_file):
            os.remove(baseline_file)
            print(f"🗑️ 强制重新开始，删除现有结果文件: {baseline_file}")
        print("🆕 将从头开始处理所有字符")
    
    # 检查是否已有结果文件，支持断点续传
    processed_characters = set()
    start_index = 0
    
    # 检查baseline文件是否存在，如果存在则读取已处理的字符
    if os.path.exists(baseline_file):
        try:
            df = pd.read_csv(baseline_file)
            if len(df) > 0:
                # 过滤掉表头行，只统计实际数据
                data_rows = df[df['Character'] != 'Character']  # 排除表头
                if len(data_rows) > 0:
                    processed_chars = set(data_rows['Character'].tolist())
                    processed_characters.update(processed_chars)
                    print(f"✅ 找到已处理的结果文件: {baseline_file}，包含 {len(processed_chars)} 个字符")
                else:
                    print(f"⚠️ 文件存在但只有表头: {baseline_file}")
        except Exception as e:
            print(f"⚠️ 读取文件失败: {baseline_file}, 错误: {e}")
    
    # 计算需要跳过的字符数量
    if processed_characters:
        print(f"📊 已处理字符: {len(processed_characters)} 个")
        print(f"📊 剩余字符: {len(All_zi) - len(processed_characters)} 个")
        
        # 找到第一个未处理的字符索引
        all_characters = All_zi['Character'].tolist()
        for i, char in enumerate(all_characters):
            if char not in processed_characters:
                start_index = i
                break
        else:
            print("🎉 所有字符都已处理完成！")
            return
    else:
        print("🆕 开始全新处理，创建新的结果文件")
        # 创建CSV文件并写入表头
        with open(baseline_file, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(["Character", "Ground_Truth", "LLM_Output", "Pipeline"])
    
    # 读取ground truth数据（使用中文释义数据）
    gt_df = pd.read_csv(str((Path(__file__).resolve().parents[1] / 'data' / 'character_explanations_CN.csv')))
    
    # 处理每个字符
    all_characters = All_zi['Character'].tolist()
    
    # 更新processed_characters集合，确保循环中的检查正确
    if processed_characters:
        print(f"🔄 断点续传模式：从第 {start_index + 1} 个字符开始")
    else:
        print(f"🆕 全新开始模式：从第 1 个字符开始")
    
    print(f"🚀 从第 {start_index + 1} 个字符开始处理，共 {len(all_characters)} 个字符")
    
    for i, zi in enumerate(all_characters):
        # 断点续传：跳过已处理的字符
        if zi in processed_characters:
            print(f"  ⏭️ 跳过已处理的字符: {zi}")
            continue
        print(f"\n------------处理字符 {i+1}/{len(all_characters)}: {zi}")
        
        img_zi_dir = str((Path(__file__).resolve().parents[1] / 'data' / 'img_zi' / zi))
        if not os.path.exists(img_zi_dir):
            print(f"❌ 字符 {zi} 的图片目录不存在: {img_zi_dir}")
            continue
        
        cnt = 0  # 每个字最多处理5张图片
        for file in os.listdir(img_zi_dir):
            if cnt >= 5:
                break
                
            if not file.endswith('.jpg'):
                continue
                
            image_name = file[:-4]  # 去掉.jpg后缀
            image_path = os.path.join(img_zi_dir, file)
            
            print(f"  处理图片: {file}")
            
            # 获取部首图像
            radical_images, radical_image_paths, target_radical = get_radical_images_from_img_zi(zi, image_name)
            
            if not radical_images:
                print(f"  跳过 {file} - 没有找到部首图像")
                continue
            
            print(f"  找到 {len(radical_images)} 个部首: {target_radical}")
            
            # 获取ground truth
            gt_row = gt_df[gt_df['Character'] == zi]
            if len(gt_row) == 0:
                print(f"  跳过 {zi} - 在ground truth中未找到")
                continue
                
            ground_truth = gt_row.iloc[0]['Explanation']
            print(f"  Ground Truth: {ground_truth}")
            
            # Baseline Pipeline - LLM获得图片信息（包括部首图像），但没有部首预测
            print(f"  🔄 Baseline Pipeline: LLM获得图片信息（包括部首图像），但没有部首预测...")
            baseline_output_cleaned = None
            
            # 为Baseline pipeline提供简洁的提示词（不包含字符名称）
            baseline_prompt = f"""
分析这个甲骨文字符的图像特征，直接输出其释义。

要求：
1. 直接输出字符的释义，不要加任何解释、分析过程或格式标记
2. 基于视觉分析给出具体含义
3. 输出格式：先给出简洁释义，然后分析象形特征

示例格式：
鬼魅。象鬼怪飄忽不定之形，表示神秘莫測的事物。
採摘。象手在樹上採摘果實之形，表示採集動作。

请直接输出这个字符的释义："""
            
            # 单次尝试，不重试
            try:
                print(f"    🔄 Baseline 处理中...")
                baseline_output, _ = chat_with_gpt_variant_explanation(
                    image_path, radical_image_paths, [], baseline_prompt, is_baseline=True  # 传入部首图像，但不传入部首预测列表，标记为Baseline
                )
                baseline_output_cleaned = clean_llm_output(baseline_output, zi, is_baseline=True)
                print(f"    ✅ Baseline输出完成: {baseline_output_cleaned}")
                        
            except Exception as e:
                print(f"    ❌ Baseline 处理失败: {e}")
                baseline_output_cleaned = ""
            
            # 保存baseline结果
            print(f"  💾 保存baseline结果...")
            
            # Baseline结果
            with open(baseline_file, mode='a', newline='', encoding='utf-8-sig', errors='ignore') as file:
                writer = csv.writer(file)
                writer.writerow([zi, ground_truth, baseline_output_cleaned, "Baseline"])
            
            # 更新已处理字符集合，用于断点续传
            processed_characters.add(zi)
            
            # 显示进度
            remaining = len(all_characters) - len(processed_characters)
            print(f"  ✅ 字符 {zi} 处理完成，剩余 {remaining} 个字符")
            
            cnt += 1

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='甲骨文解释系统 - Baseline Pipeline')
    parser.add_argument('--shuffle', action='store_true', default=True, 
                       help='是否打乱甲骨字顺序 (默认: True)')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                       help='不打乱甲骨字顺序')
    parser.add_argument('--llm-model', type=str, default=None, help='LLM 模型名')
    parser.add_argument('--llm-base-url', type=str, default=None, help='LLM API 基础URL')
    parser.add_argument('--llm-api-key', type=str, default=None, help='LLM API 密钥')
    parser.add_argument('--llm-temperature', type=float, default=None, help='LLM 温度参数')
    parser.add_argument('--llm-max-tokens', type=int, default=None, help='LLM 最大token数')
    parser.add_argument('--llm-enable-thinking', action='store_true', help='启用 thinking 模式（仅支持 reasoning 模型）')
    parser.add_argument('--llm-auto-downgrade', action='store_true', default=True, help='当模型不支持 thinking 时自动降级（默认启用）')
    parser.add_argument('--llm-no-auto-downgrade', action='store_true', help='禁用自动降级，强制使用 thinking 模式')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='训练集比例，用于构建KG (默认: 0.7)')
    parser.add_argument('--random-seed', type=int, default=42, help='随机种子，确保数据分割可重复性 (默认: 42)')
    parser.add_argument('--test-file', type=str, default=None, help='指定测试集文件路径，如果未指定则自动查找')
    parser.add_argument('--force-restart', action='store_true', help='强制重新开始，删除现有结果文件')
    parser.add_argument('--resume', action='store_true', default=True, help='断点续传模式 (默认: True)')
    
    args = parser.parse_args()
    
    # 运行期覆盖环境变量，供 chatgpt_rag.get_llm() 使用
    if args.llm_model:
        os.environ["LLM_MODEL"] = args.llm_model
    if args.llm_base_url:
        os.environ["LLM_BASE_URL"] = args.llm_base_url
    if args.llm_api_key:
        os.environ["LLM_API_KEY"] = args.llm_api_key
    if args.llm_temperature:
        os.environ["LLM_TEMPERATURE"] = str(args.llm_temperature)
    if args.llm_max_tokens:
        os.environ["LLM_MAX_TOKENS"] = str(args.llm_max_tokens)
    if args.llm_enable_thinking:
        os.environ["LLM_ENABLE_THINKING"] = "true"
    if args.llm_no_auto_downgrade:
        os.environ["LLM_AUTO_DOWNGRADE"] = "false"
    
    # 确保GPU结果一致性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 数据分割：7/10用于构建KG，3/10作为测试集
    print('📊 开始数据分割...')
    csv_file = str((Path(__file__).resolve().parents[1] / 'data' / 'character_explanations_CN.csv'))
    
    # 检查是否指定了测试集文件
    if args.test_file:
        test_file = args.test_file
        print(f'✅ 使用指定的测试集文件: {test_file}')
        test_df = pd.read_csv(test_file)
        print(f'   测试集数量: {len(test_df)} 个字符')
    else:
        # 检查是否存在已分割的测试集文件
        test_file = csv_file.replace('.csv', '_unseen.csv')
        if os.path.exists(test_file):
            print(f'✅ 找到已分割的测试集文件: {test_file}')
            test_df = pd.read_csv(test_file)
            print(f'   测试集数量: {len(test_df)} 个字符')
        else:
            print(f'⚠️ 未找到已分割的测试集文件，重新进行数据分割')
            # 使用与run_prototype_kg.py相同的数据分割逻辑
            train_df, test_df = split_data_for_kg_and_test(csv_file, train_ratio=args.train_ratio, random_seed=args.random_seed)
            print(f'   训练集数量: {len(train_df)} 个字符')
            print(f'   测试集数量: {len(test_df)} 个字符')
    
    print('⏭️ Baseline pipeline不需要知识图谱构建和缓存预热')
    
    # 检查img_zi目录
    if not os.path.exists(str((Path(__file__).resolve().parents[1] / 'data' / 'img_zi'))):
        print("data/img_zi 目录不存在")
        exit(1)
    
    # 处理测试集字符 - 仅运行baseline pipeline
    print('🚀 开始处理测试集字符 - Baseline Pipeline...')
    process_test_characters_baseline_only(test_df, force_restart=args.force_restart)
    
    print(f'\n🎉 所有字符处理完成!')
    
    # 输出结果统计
    output_dir = 'baseoutput'
    # baseline的结果文件
    baseline_file = f'{output_dir}/test_set_baseline.csv'
    
    print("🎉 Baseline pipeline的结果已保存到baseoutput目录")
    print(f"📊 测试集字符数: {len(test_df)}")
    
    # 统计baseline的结果数量
    if os.path.exists(baseline_file):
        df = pd.read_csv(baseline_file)
        print(f"📊 Baseline Pipeline总样本数: {len(df)}")
    else:
        print(f"❌ Baseline Pipeline结果文件不存在: {baseline_file}")
    
    print(f"\n📁 输出文件:")
    print(f"  Baseline: {baseline_file}")
