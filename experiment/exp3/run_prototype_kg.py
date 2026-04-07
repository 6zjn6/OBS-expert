from KG_construct import KG_construct_new
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
    chat_with_gpt_variant_explanation, 
    warm_up_cache, 
    get_cache_stats,
    search_exact_character,
    search_character_by_radical,
    search_radical_explanation,
    search_variant_characters,
    search_character_by_modern_character
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

def build_kg_with_training_data(train_df):
    """
    使用训练集数据构建知识图谱
    """
    print("🔨 使用训练集数据构建知识图谱...")
    
    # 创建临时seen数据文件（基于文件位置解析）
    base_data_dir = Path(__file__).resolve().parents[1] / 'data'
    temp_seen_file = str(base_data_dir / 'character_explanations_CN_temp_seen.csv')
    train_df.to_csv(temp_seen_file, index=False, encoding='utf-8-sig')
    
    try:
        # 临时修改环境变量，让KG构建使用seen数据
        original_default = str(base_data_dir / 'character_explanations_CN.csv')
        original_csv = os.environ.get('CHARACTER_CSV_FILE', original_default)
        os.environ['CHARACTER_CSV_FILE'] = temp_seen_file
        
        # 构建KG
        KG_construct_new()
        print("✅ 知识图谱构建成功!")
        
        # 恢复原始环境变量
        os.environ['CHARACTER_CSV_FILE'] = original_csv
        
    except Exception as e:
        print(f"❌ 知识图谱构建失败: {e}")
        # 恢复原始环境变量
        os.environ['CHARACTER_CSV_FILE'] = original_csv
        raise e
    finally:
        # 清理临时文件
        if os.path.exists(temp_seen_file):
            os.remove(temp_seen_file)

def get_possible_radical(radical_image, class_prototypes, train_classes, model, all_radical_list, std, mean, topk=5):
    """使用PrototypeClassifier预测部首"""
    return get_possible_radical_prototype(radical_image, class_prototypes, train_classes, model, all_radical_list, std, mean, topk)

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

def generate_explanation_from_database(character, radical_list):
    """基于数据库搜索生成字符释义，充分利用所有可用的搜索工具"""
    print(f"    🔍 开始数据库搜索，字符: {character}, 部首: {radical_list}")
    
    # 1. 首先尝试通过部首搜索（这是最安全的方式）
    if radical_list:
        radical_results = []
        
        for radical in radical_list:
            print(f"    🔍 搜索部首: {radical}")
            # 搜索部首解释
            radical_explanation = search_radical_explanation(radical)
            print(f"    📝 部首解释结果: {radical_explanation[:100]}...")
            if "知识库中暂无" not in radical_explanation and "查询失败" not in radical_explanation:
                radical_results.append(f"部首'{radical}': {radical_explanation}")
                print(f"    ✅ 找到部首解释: {radical}")
            
            # 搜索包含该部首的字符
            character_by_radical = search_character_by_radical(radical)
            print(f"    📝 相关字符结果: {character_by_radical[:100]}...")
            if "知识库中暂无" not in character_by_radical and "查询失败" not in character_by_radical:
                radical_results.append(f"包含部首'{radical}'的字符: {character_by_radical}")
                print(f"    ✅ 找到相关字符: {radical}")
        
        if radical_results:
            combined_result = "基于部首分析:\n" + "\n".join(radical_results)
            print(f"    ✅ 数据库搜索成功，找到 {len(radical_results)} 条信息")
            return combined_result
        else:
            print(f"    ⚠️ 部首搜索未找到有效信息")
    
    # 2. 尝试搜索变体字符
    print(f"    🔍 尝试搜索变体字符...")
    try:
        variant_result = search_variant_characters(character)
        print(f"    📝 变体字符结果: {variant_result[:100]}...")
        if "知识库中暂无" not in variant_result and "查询失败" not in variant_result:
            print(f"    ✅ 找到变体字符信息")
            return f"基于变体字符分析:\n{variant_result}"
        else:
            print(f"    ⚠️ 变体字符搜索未找到有效信息")
    except Exception as e:
        print(f"    ❌ 变体字符搜索失败: {e}")
    
    # 3. 尝试通过现代字符搜索
    print(f"    🔍 尝试搜索现代字符...")
    try:
        modern_result = search_character_by_modern_character(character)
        print(f"    📝 现代字符结果: {modern_result[:100]}...")
        if "知识库中暂无" not in modern_result and "查询失败" not in modern_result:
            print(f"    ✅ 找到现代字符信息")
            return f"基于现代字符分析:\n{modern_result}"
        else:
            print(f"    ⚠️ 现代字符搜索未找到有效信息")
    except Exception as e:
        print(f"    ❌ 现代字符搜索失败: {e}")
    
    # 4. 如果都没有找到，返回空字符串让LLM自己处理
    print(f"    ⚠️ 数据库搜索未找到有效信息，返回空字符串让LLM自己处理")
    return ""

def process_test_characters_two_pipelines(test_df, force_restart=False):
    """处理测试集字符，使用两个不同的pipeline生成甲骨字释义，支持断点续传"""
    
    # 获得PrototypeClassifier模型
    print('----------获得PrototypeClassifier模型-----------')
    model, class_prototypes, train_classes, std, mean = get_prototype_model()
    if model is None:
        print("❌ 无法获取PrototypeClassifier模型，退出")
        return
    
    model.eval()
    
    # 获取所有部首列表
    try:
        radical_df_all = pd.read_csv(str(base_data_dir / 'radical_explanation.csv'))
    except pd.errors.ParserError as e:
        print(f"⚠️ CSV解析错误: {e}")
        print("🔧 尝试使用错误处理...")
        try:
            # 尝试使用更宽松的参数
            radical_df_all = pd.read_csv(str(base_data_dir / 'radical_explanation.csv'), 
                                       on_bad_lines='skip',
                                       encoding='utf-8-sig',
                                       quoting=csv.QUOTE_NONE,
                                       error_bad_lines=False)
        except Exception as e2:
            print(f"⚠️ 第二次尝试失败: {e2}")
            print("🔧 尝试手动读取CSV文件...")
            # 手动读取CSV文件
            import csv
            rows = []
            with open(str(base_data_dir / 'radical_explanation.csv'), 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)  # 读取标题行
                for i, row in enumerate(reader, 1):
                    if len(row) == 4:  # 确保有4个字段
                        rows.append(row)
                    else:
                        print(f"⚠️ 跳过格式错误的行 {i}: {row}")
            radical_df_all = pd.DataFrame(rows, columns=header)
    
    all_radical_list = radical_df_all['Radical'].unique().tolist()
    
    # 使用测试集数据
    All_zi = test_df
    print(f"成功读取测试集数据: {len(All_zi)} 个字符")
    
    # 创建输出目录
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取大模型型号用于文件名，与chatgpt_rag.py中的get_llm()保持一致
    # 两个pipeline的CSV文件路径
    prototype_file = f'{output_dir}/test_set_prototype.csv'
    kg_file = f'{output_dir}/test_set_kg.csv'
    
    # 检查是否强制重新开始
    if force_restart:
        for file_path in [prototype_file, kg_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ 强制重新开始，删除现有结果文件: {file_path}")
        print("🆕 将从头开始处理所有字符")
    
    # 检查是否已有结果文件，支持断点续传
    processed_characters = set()
    start_index = 0
    
    # 检查两个文件是否存在，如果存在则读取已处理的字符
    for file_path in [prototype_file, kg_file]:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if len(df) > 0:
                    # 过滤掉表头行，只统计实际数据
                    data_rows = df[df['Character'] != 'Character']  # 排除表头
                    if len(data_rows) > 0:
                        processed_chars = set(data_rows['Character'].tolist())
                        processed_characters.update(processed_chars)
                        print(f"✅ 找到已处理的结果文件: {file_path}，包含 {len(processed_chars)} 个字符")
                    else:
                        print(f"⚠️ 文件存在但只有表头: {file_path}")
            except Exception as e:
                print(f"⚠️ 读取文件失败: {file_path}, 错误: {e}")
    
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
        # 创建两个CSV文件并写入表头
        for file_path in [prototype_file, kg_file]:
            with open(file_path, mode='w', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file)
                writer.writerow(["Character", "Ground_Truth", "LLM_Output", "Pipeline"])
    
    # 读取ground truth数据（使用中文释义数据）
    gt_df = pd.read_csv(str(base_data_dir / 'character_explanations_CN.csv'))
    
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
            
            # 获取部首图像
            radical_images, radical_image_paths, target_radical = get_radical_images_from_img_zi(zi, image_name)
            
            if not radical_images:
                print(f"  跳过 {file} - 没有找到部首图像")
                continue
            
            # 使用PrototypeClassifier预测可能的部首
            possible_radicals = []
            best_radicals = []  # 只保留每个部首图像中最可能的部首
            
            for radical_image in radical_images:
                possible_radical = get_possible_radical(radical_image, class_prototypes, train_classes, 
                                                      model, all_radical_list, std, mean)
                possible_radicals.append(possible_radical)
                
                # 只取相似度最高的部首
                if possible_radical and len(possible_radical) > 0:
                    best_radical = possible_radical[0]  # 取第一个（相似度最高）
                    best_radicals.append(best_radical)
            
            print(f"  部首: {best_radicals}")
            
            # 获取ground truth
            gt_row = gt_df[gt_df['Character'] == zi]
            if len(gt_row) == 0:
                print(f"  跳过 {zi} - 在ground truth中未找到")
                continue
                
            ground_truth = gt_row.iloc[0]['Explanation']
            
            # Pipeline 1: Prototype - 只使用部首预测，不调用数据库
            print(f"  🔄 Pipeline 1 (Prototype): 只使用部首预测，不调用数据库...")
            prototype_output_cleaned = None
            
            # 为Prototype pipeline提供简洁的提示词
            prototype_prompt = f"""
分析这个甲骨文字符的图像特征和部首信息：{best_radicals}，直接输出其释义。

要求：
1. 直接输出字符的释义，不要加任何解释、分析过程或格式标记
2. 基于视觉分析和部首信息给出具体含义
3. 输出格式：先给出简洁释义，然后分析部首构成和象形特征

示例格式：
鬼魅。從鬼從彡，象鬼怪飄忽不定之形，彡表示其光影閃爍。
採摘。從爪從木，象手在樹上採摘果實之形。

请直接输出这个字符的释义："""
            
            # 单次尝试，不重试
            try:
                print(f"    🔄 Prototype 处理中...")
                prototype_output, _ = chat_with_gpt_variant_explanation(
                    image_path, radical_image_paths, best_radicals, prototype_prompt, False, True
                )
                prototype_output_cleaned = clean_llm_output(prototype_output, zi, is_baseline=False)
                print(f"    ✅ Prototype输出完成")
                        
            except Exception as e:
                print(f"    ❌ Prototype 处理失败: {e}")
                prototype_output_cleaned = ""
            
            # Pipeline 2: KG - 使用部首预测 + 数据库搜索
            print(f"  🔄 Pipeline 2 (KG): 使用部首预测 + 数据库搜索...")
            kg_output_cleaned = None
            
            # 使用数据库搜索
            print(f"    🔍 数据库搜索...")
            database_output = generate_explanation_from_database(zi, best_radicals)
            print(f"    📊 数据库搜索结果长度: {len(database_output)} 字符")
            print(f"    📊 数据库搜索结果预览: {database_output[:200]}...")
            
            # 将数据库信息作为上下文传递给LLM
            enhanced_prompt = f"""
数据库信息：{database_output}

基于图像分析、部首信息{best_radicals}和数据库信息，直接输出这个甲骨文字符的释义。

要求：
1. 直接输出字符的释义，不要加任何解释、分析过程或格式标记
2. 基于视觉分析给出具体含义
3. 输出格式：先给出简洁释义，然后分析部首构成和象形特征

示例格式：
鬼魅。從鬼從彡，象鬼怪飄忽不定之形，彡表示其光影閃爍。
採摘。從爪從木，象手在樹上採摘果實之形。

请直接输出这个字符的释义："""
            
            # 单次尝试，不重试
            try:
                print(f"    🔄 KG 处理中...")
                kg_output, _ = chat_with_gpt_variant_explanation(
                    image_path, radical_image_paths, best_radicals, enhanced_prompt, False, True
                )
                kg_output_cleaned = clean_llm_output(kg_output, zi, is_baseline=False)
                print(f"    ✅ KG输出完成")
                        
            except Exception as e:
                print(f"    ❌ KG 处理失败: {e}")
                kg_output_cleaned = ""
            
            # 保存两个pipeline的结果
            with open(prototype_file, mode='a', newline='', encoding='utf-8-sig', errors='ignore') as file:
                writer = csv.writer(file)
                writer.writerow([zi, ground_truth, prototype_output_cleaned, "Prototype"])
            
            with open(kg_file, mode='a', newline='', encoding='utf-8-sig', errors='ignore') as file:
                writer = csv.writer(file)
                writer.writerow([zi, ground_truth, kg_output_cleaned, "KG"])
            
            # 更新已处理字符集合，用于断点续传
            processed_characters.add(zi)
            
            # 显示进度
            remaining = len(all_characters) - len(processed_characters)
            print(f"  ✅ 字符 {zi} 处理完成，剩余 {remaining} 个字符")
            
            cnt += 1

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='甲骨文解释系统 - Prototype + KG Pipeline')
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
    parser.add_argument('--force-rebuild-kg', action='store_true', default=True, help='强制重建知识图谱，删除现有内容 (默认: True)')
    parser.add_argument('--no-force-rebuild-kg', dest='force_rebuild_kg', action='store_false', help='不强制重建知识图谱，如果存在则跳过')
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
    train_df, test_df = split_data_for_kg_and_test(csv_file, train_ratio=args.train_ratio, random_seed=args.random_seed)
    
    # 根据命令行参数决定是否强制重建知识图谱
    if args.force_rebuild_kg:
        print('🗑️ 强制删除现有知识图谱...')
        try:
            from py2neo import Graph
            graph = Graph("bolt://localhost:7687", auth=("neo4j", "87654321"))
            
            # 检查现有节点数量
            node_count = len(graph.nodes)
            if node_count > 0:
                print(f'⚠️ 发现现有知识图谱，包含 {node_count} 个节点')
                print('🗑️ 正在清空知识图谱...')
                
                # 删除所有节点和关系
                graph.run("MATCH (n) DETACH DELETE n")
                print('✅ 知识图谱已完全清空')
            else:
                print('ℹ️ 知识图谱为空，无需清空')
                
        except Exception as e:
            print(f'⚠️ 清空知识图谱失败: {e}')
            print('🔨 继续构建新知识图谱...')
        
        # 构建新的知识图谱
        print('🔨 开始构建新的知识图谱...')
        build_kg_with_training_data(train_df)
    else:
        # 检查知识图谱是否已存在，如果存在则跳过构建
        print('🔍 检查现有知识图谱...')
        try:
            from py2neo import Graph
            graph = Graph("bolt://localhost:7687", auth=("neo4j", "87654321"))
            node_count = len(graph.nodes)
            if node_count > 0:
                print(f'✅ 知识图谱已存在，包含 {node_count} 个节点，跳过构建')
                print('⚠️ 注意：使用现有KG可能导致数据泄露！')
            else:
                print('🔨 KG构建开始!')
                # 使用训练集数据构建KG
                build_kg_with_training_data(train_df)
        except Exception as e:
            print(f'⚠️ 检查知识图谱失败: {e}')
            print('🔨 KG构建开始!')
            # 使用训练集数据构建KG
            build_kg_with_training_data(train_df)
    
    # 预热缓存（只预热一次）
    print('🔥 缓存预热开始!')
    warm_up_cache()
    print('✅ 缓存预热完成!')
    
    # 检查img_zi目录
    if not os.path.exists(str((Path(__file__).resolve().parents[1] / 'data' / 'img_zi'))):
        print("data/img_zi 目录不存在")
        exit(1)
    
    # 处理测试集字符
    print('🚀 开始处理测试集字符...')
    process_test_characters_two_pipelines(test_df, force_restart=args.force_restart)
    
    print(f'\n🎉 所有字符处理完成!')
    
    # 输出缓存统计
    print('\n📊 缓存统计信息:')
    cache_stats = get_cache_stats()
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")
    
    # 输出结果统计
    output_dir = 'exp2_output'

    # 两个pipeline的结果文件
    prototype_file = f'{output_dir}/test_set_prototype.csv'
    kg_file = f'{output_dir}/test_set_kg.csv'
    
    print("🎉 两个pipeline的结果已保存到exp2_output目录")
    print(f"📊 训练集字符数: {len(train_df)}")
    print(f"📊 测试集字符数: {len(test_df)}")
    
    # 统计各pipeline的结果数量
    for file_path, pipeline_name in [(prototype_file, "Prototype"), (kg_file, "KG")]:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"📊 {pipeline_name} Pipeline总样本数: {len(df)}")
        else:
            print(f"❌ {pipeline_name} Pipeline结果文件不存在: {file_path}")
    
    print(f"\n📁 输出文件:")
    print(f"  Prototype: {prototype_file}")
    print(f"  KG: {kg_file}")
