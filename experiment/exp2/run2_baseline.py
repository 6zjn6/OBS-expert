from KG_construct import KG_construct_new
import itertools
import pandas as pd
import os
import shutil
import tempfile
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import numpy as np
from torchvision import transforms
import torch
import csv
from sklearn.model_selection import train_test_split
import re
from config import get_prototype_model, get_possible_radical_prototype
from chatgpt_rag import chat_with_gpt_new_noimage_wrapper, chat_with_gpt_new_bothimage_wrapper, warm_up_cache, \
    get_cache_stats
import argparse
from pathlib import Path

# 全局变量
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_possible_radical(radical_image, class_prototypes, train_classes, model, all_radical_list, std, mean, topk=5):
    """使用PrototypeClassifier预测部首"""
    return get_possible_radical_prototype(radical_image, class_prototypes, train_classes, model, all_radical_list, std,
                                          mean, topk)


def create_anonymous_image_paths(image_path, radical_image_paths):
    """创建匿名图片路径，避免LLM从文件名获取字符信息"""
    temp_dir = tempfile.mkdtemp(prefix="anonymous_images_")
    anonymous_paths = []
    
    # 复制主图片
    main_ext = os.path.splitext(image_path)[1]
    anonymous_main_path = os.path.join(temp_dir, f"main_image{main_ext}")
    shutil.copy2(image_path, anonymous_main_path)
    anonymous_paths.append(anonymous_main_path)
    
    # 复制部首图片
    anonymous_radical_paths = []
    for i, radical_path in enumerate(radical_image_paths):
        radical_ext = os.path.splitext(radical_path)[1]
        anonymous_radical_path = os.path.join(temp_dir, f"radical_{i}{radical_ext}")
        shutil.copy2(radical_path, anonymous_radical_path)
        anonymous_radical_paths.append(anonymous_radical_path)
    
    return anonymous_main_path, anonymous_radical_paths, temp_dir


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


def fix_csv_format_errors(csv_file_path):
    """修复CSV文件中的格式错误"""
    if not os.path.exists(csv_file_path):
        return
    
    print(f"🔧 检查并修复CSV文件格式错误: {csv_file_path}")
    
    try:
        df = pd.read_csv(csv_file_path)
        valid_types = ['象形字', '会意字', '形声字']
        
        # 检查Predicted_Type列中的无效值
        invalid_mask = ~df['Predicted_Type'].isin(valid_types + ['解析失败'])
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            print(f"  ⚠️  发现 {invalid_count} 条无效的预测类型记录")
            
            # 修复无效记录
            for idx in df[invalid_mask].index:
                original_type = df.loc[idx, 'Predicted_Type']
                print(f"    🔧 修复记录 {idx}: {original_type} -> 解析失败")
                
                df.loc[idx, 'Predicted_Type'] = "解析失败"
                df.loc[idx, 'Predicted_Reasoning'] = f"原始输出解析失败: {original_type}"
                df.loc[idx, 'Type_Correct'] = 0
            
            # 保存修复后的文件
            df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
            print(f"  ✅ 已修复并保存CSV文件")
        else:
            print(f"  ✅ CSV文件格式正确，无需修复")
            
    except Exception as e:
        print(f"  ❌ 修复CSV文件时出错: {e}")


def process_all_characters_baseline():
    """处理所有字符的Baseline评估"""

    # 获得PrototypeClassifier模型
    print('----------获得PrototypeClassifier模型-----------')
    model, class_prototypes, train_classes, std, mean = get_prototype_model()
    if model is None:
        print("❌ 无法获取PrototypeClassifier模型，退出")
        return

    model.eval()

    # 获取所有部首列表（基于文件位置解析）
    base_data_dir = Path(__file__).resolve().parents[1] / 'data'
    radical_df_all = pd.read_csv(str(base_data_dir / 'radical_explanation.csv'))
    all_radical_list = radical_df_all['Radical'].unique().tolist()

    # 读取所有字符数据
    try:
        All_zi = pd.read_csv(str(base_data_dir / 'character_analysis.csv'))
        print(f"成功读取字符数据: {len(All_zi)} 个字符")
    except Exception as e:
        print(f"读取字符数据失败: {e}")
        return

    # 创建输出目录
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # CSV文件路径
    baseline_csv_file = f'{output_dir}/final_output_baseline.csv'
    
    # 修复已存在的CSV文件中的格式错误
    fix_csv_format_errors(baseline_csv_file)

    # 创建CSV文件并写入表头（如果文件不存在）
    if not os.path.exists(baseline_csv_file):
        with open(baseline_csv_file, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Character", "Ground_Truth_Type", "Ground_Truth_Reasoning", "Predicted_Type", "Predicted_Reasoning",
                 "Type_Correct"])
        print(f"📝 创建新的CSV文件: {baseline_csv_file}")
    else:
        print(f"📝 使用已存在的CSV文件: {baseline_csv_file}")

    # 读取ground truth数据
    gt_df = pd.read_csv(str(base_data_dir / 'character_analysis.csv'))

    def get_processed_characters():
        """获取已经处理过的字符列表"""
        processed_chars = set()
        
        # 检查baseline文件
        if os.path.exists(baseline_csv_file):
            try:
                baseline_df = pd.read_csv(baseline_csv_file)
                if len(baseline_df) > 0:
                    processed_chars.update(baseline_df['Character'].tolist())
                    print(f"📋 从baseline文件找到 {len(baseline_df)} 条已处理记录")
            except Exception as e:
                print(f"⚠️  读取baseline文件失败: {e}")
        
        return processed_chars

    def get_processed_images_for_character(character):
        """获取指定字符已处理的图片数量"""
        processed_count = 0
        
        # 检查baseline文件
        if os.path.exists(baseline_csv_file):
            try:
                baseline_df = pd.read_csv(baseline_csv_file)
                if len(baseline_df) > 0:
                    char_records = baseline_df[baseline_df['Character'] == character]
                    processed_count = max(processed_count, len(char_records))
            except Exception as e:
                pass
        
        return processed_count

    def parse_llm_output(output_text):
        """解析LLM的简洁格式输出"""
        character_type = ""
        reasoning = ""

        if not output_text or output_text.strip() == "":
            print(f"    ⚠️  LLM输出为空，使用默认值")
            character_type = '象形字'
            reasoning = '象形之形'
            print(f"    🔧 使用默认值: {character_type} - {reasoning}")
            return character_type, reasoning

        print(f"    📝 原始输出: {output_text[:200]}...")
        
        # 定义标准的字符类型
        valid_types = ['象形字', '会意字', '形声字']
        
        # 解析新的简单格式
        lines = output_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('类型：'):
                candidate_type = line.replace('类型：', '').strip()
                # 验证是否为有效的字符类型
                if any(valid_type in candidate_type for valid_type in valid_types):
                    character_type = candidate_type
            elif line.startswith('推理：'):
                reasoning = line.replace('推理：', '').strip()

        # 如果新格式解析失败，尝试旧格式
        if not character_type or not reasoning:
            print(f"    ⚠️  新格式解析失败，尝试旧格式")
            
            # 尝试旧格式解析
            for line in lines:
                line = line.strip()
                if line.startswith('character_type:'):
                    candidate_type = line.replace('character_type:', '').strip()
                    # 验证是否为有效的字符类型
                    if any(valid_type in candidate_type for valid_type in valid_types):
                        character_type = candidate_type
                elif line.startswith('reasoning:'):
                    reasoning = line.replace('reasoning:', '').strip()

            # 如果还是没找到，尝试其他可能的格式
            if not character_type or not reasoning:
                print(f"    ⚠️  旧格式解析也失败，尝试其他格式")
                
                # 尝试查找包含"象形字"、"会意字"、"形声字"的行，但要求更严格的匹配
                for line in lines:
                    line = line.strip()
                    # 检查是否包含且只包含一个有效的字符类型
                    found_types = [valid_type for valid_type in valid_types if valid_type in line]
                    if len(found_types) == 1:
                        if not character_type:
                            # 提取包含有效类型的部分
                            character_type = found_types[0]
                        elif not reasoning:
                            reasoning = line
                        break

                # 如果还是没找到，尝试提取最后几行作为reasoning
                if not reasoning and len(lines) > 0:
                    # 取最后几行作为reasoning
                    last_lines = [line.strip() for line in lines[-3:] if line.strip()]
                    reasoning = ' '.join(last_lines)

        # 最终验证和清理
        if character_type:
            # 清理character_type中的额外文本，只保留有效的字符类型
            found_valid_types = [valid_type for valid_type in valid_types if valid_type in character_type]
            if found_valid_types:
                character_type = found_valid_types[0]  # 取第一个匹配的有效类型
                print(f"    🔧 清理后的character_type: {character_type}")
            else:
                print(f"    ⚠️  解析出的character_type无效: {character_type}")
                character_type = ""
        
        # 强制选择一个character_type，如果没有解析到则默认选择
        if not character_type:
            # 尝试从原始输出中猜测类型
            output_lower = output_text.lower()
            if '象形' in output_text or '象' in output_text:
                character_type = '象形字'
                print(f"    🔧 根据关键词猜测为: {character_type}")
            elif '会意' in output_text or '组合' in output_text or '合' in output_text:
                character_type = '会意字'
                print(f"    🔧 根据关键词猜测为: {character_type}")
            elif '形声' in output_text or '声' in output_text:
                character_type = '形声字'
                print(f"    🔧 根据关键词猜测为: {character_type}")
            else:
                # 默认选择象形字
                character_type = '象形字'
                print(f"    🔧 默认选择: {character_type}")
        
        # 清理reasoning，提取简洁的象形描述（参考ground truth格式）
        if reasoning:
            # 移除多余空格和换行符
            reasoning = ' '.join(reasoning.split())
            
            # 尝试提取象形描述模式，如"象...之形"
            import re
            pattern = r'象[^。，；！？]*?之形'
            matches = re.findall(pattern, reasoning)
            if matches:
                # 取第一个匹配的象形描述
                reasoning = matches[0]
                print(f"    🔧 提取象形描述: {reasoning}")
            else:
                # 尝试提取其他简洁模式
                # 1. 提取"象...之意"模式
                pattern2 = r'象[^。，；！？]*?之意'
                matches2 = re.findall(pattern2, reasoning)
                if matches2:
                    reasoning = matches2[0]
                    print(f"    🔧 提取象意描述: {reasoning}")
                else:
                    # 2. 提取"象...之"模式
                    pattern3 = r'象[^。，；！？]*?之[^。，；！？]*'
                    matches3 = re.findall(pattern3, reasoning)
                    if matches3:
                        reasoning = matches3[0]
                        print(f"    🔧 提取象形相关: {reasoning}")
                    else:
                        # 3. 提取数字相关描述
                        pattern4 = r'数目[^。，；！？]*|数字[^。，；！？]*'
                        matches4 = re.findall(pattern4, reasoning)
                        if matches4:
                            reasoning = matches4[0]
                            print(f"    🔧 提取数字描述: {reasoning}")
                        else:
                            # 4. 简化其他描述，移除冗余词汇
                            reasoning = re.sub(r'这个字符|该字符|这个字|该字|表示|代表|意思是|含义是|通过|方式|描绘|描述', '', reasoning)
                            reasoning = re.sub(r'[。，；！？]', '', reasoning)
                            reasoning = reasoning.strip()
                            # 如果还是太长，取前20个字符
                            if len(reasoning) > 20:
                                reasoning = reasoning[:20]
                            print(f"    🔧 简化推理: {reasoning}")
        
        # 如果reasoning为空，提供简洁的默认值（参考ground truth格式）
        if not reasoning:
            if character_type == '象形字':
                reasoning = '象形之形'
            elif character_type == '会意字':
                reasoning = '会意组合'
            elif character_type == '形声字':
                reasoning = '形声结构'
            else:
                reasoning = '字形分析'
            print(f"    🔧 使用默认推理: {reasoning}")

        # 最终检查 - character_type现在必须有值
        if not character_type:
            character_type = '象形字'  # 最后的保险
            print(f"    ⚠️  强制设置character_type为: {character_type}")

        print(f"    ✅ 解析结果: {character_type} - {reasoning}")
        return character_type, reasoning

    # 获取已处理的字符
    print("🔍 检查已处理的字符...")
    processed_chars = get_processed_characters()
    
    # 处理每个字符
    all_characters = All_zi['character'].tolist()
    print(f"📊 总字符数: {len(all_characters)}")
    print(f"📊 已处理字符数: {len(processed_chars)}")
    
    # 过滤出未处理的字符
    remaining_chars = [char for char in all_characters if char not in processed_chars]
    print(f"📊 剩余待处理字符数: {len(remaining_chars)}")
    
    if len(remaining_chars) == 0:
        print("✅ 所有字符都已处理完成！")
        return
    
    print(f"🚀 开始处理剩余字符...")

    for i, zi in enumerate(remaining_chars):
        total_processed = len(processed_chars) + i + 1
        print(f"\n------------处理字符 {total_processed}/{len(all_characters)}: {zi} (剩余: {len(remaining_chars) - i - 1})")

        img_zi_dir = str((Path(__file__).resolve().parents[1] / 'data' / 'img_zi' / zi))
        if not os.path.exists(img_zi_dir):
            print(f"❌ 字符 {zi} 的图片目录不存在: {img_zi_dir}")
            continue

        # 检查该字符已处理的图片数量
        processed_images = get_processed_images_for_character(zi)
        if processed_images >= 5:
            print(f"  ⏭️  字符 {zi} 已处理完所有图片 ({processed_images}/5)，跳过")
            continue
        
        print(f"  📊 字符 {zi} 已处理 {processed_images}/5 张图片，继续处理...")
        
        cnt = processed_images  # 从已处理的数量开始
        image_files = [f for f in os.listdir(img_zi_dir) if f.endswith('.jpg')]
        
        for file in image_files:
            if cnt >= 5:
                break

            image_name = file[:-4]  # 去掉.jpg后缀
            image_path = os.path.join(img_zi_dir, file)

            print(f"  处理图片: {file} (第 {cnt + 1}/5 张)")

            # 获取部首图像
            radical_images, radical_image_paths, target_radical = get_radical_images_from_img_zi(zi, image_name)

            if not radical_images:
                print(f"  跳过 {file} - 没有找到部首图像")
                continue

            print(f"  找到 {len(radical_images)} 个部首: {target_radical}")

            # 创建匿名图片路径，避免LLM从文件名获取字符信息
            print(f"  🔒 创建匿名图片路径...")
            anonymous_main_path, anonymous_radical_paths, temp_dir = create_anonymous_image_paths(image_path, radical_image_paths)
            
            # 获取模型输出（带重试机制）
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    print(f"  🔄 调用Baseline... (尝试 {retry_count + 1}/{max_retries})")
                    model_output_baseline, _ = chat_with_gpt_new_bothimage_wrapper(anonymous_main_path, anonymous_radical_paths,
                                                                                   [])  # baseline不得到预测的部首列表
                    break  # 成功，跳出重试循环
                except Exception as e:
                    retry_count += 1
                    print(f"  ⚠️  调用失败 (尝试 {retry_count}/{max_retries}): {e}")
                    if retry_count >= max_retries:
                        print(f"  ❌ 达到最大重试次数，跳过此图片")
                        # 清理临时文件
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        continue
                    print(f"  🔄 等待5秒后重试...")
                    import time
                    time.sleep(5)
            
            # 清理临时文件
            shutil.rmtree(temp_dir, ignore_errors=True)

            # 解析模型输出
            pred_type_baseline, pred_reasoning_baseline = parse_llm_output(model_output_baseline)

            print(f"  Baseline预测: {pred_type_baseline} - {pred_reasoning_baseline}")

            # 获取ground truth
            gt_row = gt_df[gt_df['character'] == zi]
            if len(gt_row) == 0:
                print(f"  跳过 {zi} - 在ground truth中未找到")
                continue

            gt_type = gt_row.iloc[0]['character_type']
            gt_reasoning = gt_row.iloc[0]['reasoning']

            print(f"  Ground Truth: {gt_type} - {gt_reasoning}")

            # 计算character_type准确度
            type_correct_baseline = 1 if pred_type_baseline == gt_type else 0

            print(f"  Baseline类型准确: {type_correct_baseline}")

            # 现在不会有解析失败的情况，character_type总是有效的

            # 保存baseline结果
            with open(baseline_csv_file, mode='a', newline='', encoding='utf-8-sig', errors='ignore') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [zi, gt_type, gt_reasoning, pred_type_baseline, pred_reasoning_baseline, type_correct_baseline])

            cnt += 1
            
            # 每处理完一个字符的所有图片，显示进度
            if cnt >= 5:
                print(f"  ✅ 字符 {zi} 处理完成 ({cnt}/5 张图片)")
            
            # 每处理10个字符，显示总体进度
            if (i + 1) % 10 == 0:
                total_processed = len(processed_chars) + i + 1
                print(f"\n📊 进度报告: 已处理 {total_processed}/{len(all_characters)} 个字符 ({total_processed/len(all_characters)*100:.1f}%)")
                print(f"📊 剩余字符: {len(remaining_chars) - i - 1} 个")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='甲骨文解释系统 - Baseline Pipeline')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='是否打乱甲骨字顺序 (默认: True)')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                        help='不打乱甲骨字顺序')
    parser.add_argument('--llm-model', type=str, default=None,
                        help='LLM 模型名')
    parser.add_argument('--llm-base-url', type=str, default=None,
                        help='OpenAI 兼容的 Base URL（例如 https://openrouter.ai/api/v1）')
    parser.add_argument('--llm-api-key', type=str, default=None, help='LLM API Key')
    parser.add_argument('--llm-temperature', type=float, default=None, help='LLM 采样温度')
    parser.add_argument('--llm-max-tokens', type=int, default=None, help='LLM 最大输出 tokens')
    args = parser.parse_args()

    # 运行期覆盖环境变量，供 chatgpt_rag.get_llm() 使用
    if args.llm_model:
        os.environ['LLM_MODEL'] = args.llm_model
    if args.llm_base_url:
        os.environ['LLM_BASE_URL'] = args.llm_base_url
    if args.llm_api_key:
        os.environ['LLM_API_KEY'] = args.llm_api_key
    if args.llm_temperature is not None:
        os.environ['LLM_TEMPERATURE'] = str(args.llm_temperature)
    if args.llm_max_tokens is not None:
        os.environ['LLM_MAX_TOKENS'] = str(args.llm_max_tokens)

    # 确保GPU结果一致性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Baseline方法不需要知识图谱和缓存
    print('ℹ️  Baseline方法：跳过知识图谱构建和缓存预热')

    # 检查img_zi目录
    if not os.path.exists(str((Path(__file__).resolve().parents[1] / 'data' / 'img_zi'))):
        print("data/img_zi 目录不存在")
        exit(1)

    # 直接处理所有字符
    print('🚀 开始处理所有字符 (Baseline Pipeline)...')
    process_all_characters_baseline()

    print(f'\n🎉 所有字符处理完成!')

    # Baseline方法不使用缓存，跳过缓存统计
    print('\nℹ️  Baseline方法：跳过缓存统计')

    # 计算最终结果
    output_dir = 'output'
    baseline_file = f'{output_dir}/final_output_baseline.csv'

    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)
        baseline_accuracy = baseline_df['Type_Correct'].mean()
        print(f"📊 Baseline Character Type准确度: {baseline_accuracy:.4f} ({baseline_accuracy * 100:.2f}%)")
        print(f"📊 Baseline总样本数: {len(baseline_df)}")

    # 输出详细统计信息
    if os.path.exists(baseline_file):
        print("\n📈 按字符类型统计准确度:")
        for char_type in ['象形字', '会意字', '形声字']:
            baseline_type_acc = baseline_df[baseline_df['Ground_Truth_Type'] == char_type]['Type_Correct'].mean()
            baseline_count = len(baseline_df[baseline_df['Ground_Truth_Type'] == char_type])

            print(f"{char_type}:")
            print(f"  Baseline: {baseline_type_acc:.4f} ({baseline_count}个样本)")

    print("🎉 Baseline结果已保存到output目录")
