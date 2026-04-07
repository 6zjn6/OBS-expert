import os
import csv
from pathlib import Path
from chatgpt_rag import chat_with_gpt_variant_explanation
from config import get_prototype_model, get_possible_radical_prototype

def find_variant_characters():
    """
    找出seen和unseen数据集中重复出现的字符（异体字）
    """
    # 读取seen数据集
    seen_chars = set()
    base_data_dir = Path(__file__).resolve().parents[1] / 'data'
    with open(str(base_data_dir / 'character_explanations_CN_seen.csv'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过标题行
            if line.strip():
                parts = line.strip().split(',', 1)
                if len(parts) >= 1:
                    char = parts[0]
                    seen_chars.add(char)
    
    # 读取unseen数据集
    unseen_chars = set()
    with open(str(base_data_dir / 'character_explanations_CN_unseen.csv'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过标题行
            if line.strip():
                parts = line.strip().split(',', 1)
                if len(parts) >= 1:
                    char = parts[0]
                    unseen_chars.add(char)
    
    # 找出重复字符（异体字）
    variant_chars = seen_chars.intersection(unseen_chars)
    
    print(f"Seen数据集字符数量: {len(seen_chars)}")
    print(f"Unseen数据集字符数量: {len(unseen_chars)}")
    print(f"异体字数量: {len(variant_chars)}")
    print(f"异体字列表: {sorted(variant_chars)}")
    
    # 详细检查前几个字符
    print(f"\n详细检查前5个异体字:")
    for i, char in enumerate(sorted(variant_chars)[:5]):
        print(f"  {i+1}. {char}")
        # 检查seen数据集中的解释
        with open(str(base_data_dir / 'character_explanations_CN_seen.csv'), 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(char + ','):
                    print(f"     Seen数据集解释: {line.strip()}")
                    break
        # 检查unseen数据集中的解释
        with open(str(base_data_dir / 'character_explanations_CN_unseen.csv'), 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(char + ','):
                    print(f"     Unseen数据集解释: {line.strip()}")
                    break
    
    return sorted(variant_chars)

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
                    radical_image_paths.append(image_path)
                    
                    # 读取图像数据（用于PrototypeClassifier）
                    try:
                        from PIL import Image
                        import numpy as np
                        import cv2
                        image = Image.open(image_path)
                        # 转换为RGB格式
                        image = np.array(image)
                        if len(image.shape) == 2:
                            image = np.expand_dims(image, axis=-1)
                        if image.shape[-1] == 1:
                            image = np.repeat(image, 3, axis=-1)
                        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        radical_images.append(image)
                    except Exception as e:
                        print(f"读取部首图片失败: {e}")
                        radical_images.append(None)
            
            break
    
    print(f"  找到部首图片: {len(radical_image_paths)} 个")
    print(f"  部首名称: {target_radical}")
    print(f"  部首路径: {radical_image_paths}")
    
    return radical_images, radical_image_paths, target_radical

def get_possible_radical(radical_image, class_prototypes, train_classes, model, all_radical_list, std, mean, topk=5):
    """使用PrototypeClassifier预测部首"""
    return get_possible_radical_prototype(radical_image, class_prototypes, train_classes, model, all_radical_list, std, mean, topk)

def get_variant_prediction(image_path, radical_image_paths=None, predicted_radicals=None, max_retries=3):
    """
    上传甲骨文图片给LLM，调用RAG函数，识别对应的现代汉字
    
    Args:
        image_path: 甲骨文图片路径
        radical_image_paths: 部首图片路径列表（可选）
        predicted_radicals: 预测的部首列表（可选）
        max_retries: 最大重试次数
    
    Returns:
        str: 对应的现代汉字（如"女"、"学"等）或top5结果
    """
    if not os.path.exists(image_path):
        print(f"图片文件不存在: {image_path}")
        return None
    
    # 设置默认值
    if radical_image_paths is None:
        radical_image_paths = []
    if predicted_radicals is None:
        predicted_radicals = []
    
    # 构建改进的提示词 - 要求返回top10结果
    prompt = """你是一个汉字学专家。请分析这张甲骨文图片中的字符。

问题：这个甲骨文对应现在汉语哪个字？

要求：
1. 请给出最可能的10个现代汉字，按可能性从高到低排序
2. 格式：汉字1,汉字2,汉字3,汉字4,汉字5,汉字6,汉字7,汉字8,汉字9,汉字10
3. 10个汉字必须互不相同，不能重复
4. 如果无法识别，输出"无法识别"
5. 不要输出其他内容

示例：
输入：甲骨文图片
输出：女,母,好,如,始,妻,妇,娘,妹,姐

请直接输出10个最可能的现代汉字："""
    
    # 调试信息：显示LLM接收到的信息
    print(f"    🔍 调试信息:")
    print(f"      主图片路径: {image_path}")
    print(f"      部首图片路径: {radical_image_paths}")
    print(f"      预测部首: {predicted_radicals}")
    print(f"      提示词长度: {len(prompt)}")
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            print(f"    🔄 尝试第 {attempt + 1} 次调用...")
            
            # 调用RAG函数 - 按照run3.py的调用方式
            response, _ = chat_with_gpt_variant_explanation(
                image_path,           # 主图片路径
                radical_image_paths,  # 部首图片路径列表
                predicted_radicals,   # 预测的部首列表
                prompt               # 自定义提示词
            )
            
            # 调试信息：显示LLM的原始响应
            print(f"    🔍 LLM原始响应: '{response}'")
            print(f"    🔍 响应类型: {type(response)}")
            print(f"    🔍 响应长度: {len(response) if response else 0}")
            
            if response and response.strip():
                cleaned_response = response.strip()
                print(f"    🔍 清理后响应: '{cleaned_response}'")
                
                # 验证响应格式
                if "无法识别" in cleaned_response:
                    print(f"    ⚠️ LLM明确表示无法识别")
                    return cleaned_response
                
                # 检查是否包含多个汉字（用逗号分隔）
                if "," in cleaned_response:
                    # 解析top10结果
                    candidates = [c.strip() for c in cleaned_response.split(",")]
                    if len(candidates) >= 1:
                        print(f"    ✅ 成功获取top10结果: {candidates}")
                        return cleaned_response
                
                # 如果只有一个汉字，也接受
                if len(cleaned_response) == 1 and cleaned_response.isalpha():
                    print(f"    ✅ 成功获取单个汉字: {cleaned_response}")
                    return cleaned_response
                
                print(f"    ⚠️ 响应格式不符合预期，重试...")
            else:
                print(f"    ⚠️ LLM返回空响应，重试...")
            
        except Exception as e:
            print(f"    ❌ 第 {attempt + 1} 次调用失败: {e}")
            if attempt < max_retries - 1:
                print(f"    🔄 准备重试...")
                import time
                time.sleep(1)  # 等待1秒后重试
            else:
                import traceback
                traceback.print_exc()
    
    print(f"    ❌ 所有重试都失败，返回默认结果")
    return "预测失败"

def process_all_variant_characters():
    """
    处理所有异体字，使用部首识别模块
    """
    print("开始处理所有异体字...")
    
    # 1. 找出所有异体字
    variant_chars = find_variant_characters()
    
    if not variant_chars:
        print("没有找到异体字")
        return
    
    # 2. 获得PrototypeClassifier模型
    print('----------获得PrototypeClassifier模型-----------')
    model, class_prototypes, train_classes, std, mean = get_prototype_model()
    if model is None:
        print("❌ 无法获取PrototypeClassifier模型，退出")
        return
    
    model.eval()
    
    # 3. 获取所有部首列表
    all_radical_list = []
    with open(str(base_data_dir / 'radical_explanation.csv'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过标题行
            if line.strip():
                parts = line.strip().split(',', 1)
                if len(parts) >= 1:
                    radical = parts[0]
                    all_radical_list.append(radical)
    
    # 4. 处理每个异体字
    results = []
    
    for i, character in enumerate(variant_chars, 1):
        print(f"\n------------处理异体字 {i}/{len(variant_chars)}: {character}")
        
        img_zi_dir = str((Path(__file__).resolve().parents[1] / 'data' / 'img_zi' / character))
        if not os.path.exists(img_zi_dir):
            print(f"❌ 字符 {character} 的图片目录不存在: {img_zi_dir}")
            continue
        
        # 处理每个字符的第一张图片
        cnt = 0
        for file in os.listdir(img_zi_dir):
            if cnt >= 1:  # 每个字只处理1张图片
                break
                
            if not file.endswith('.jpg'):
                continue
                
            image_name = file[:-4]  # 去掉.jpg后缀
            image_path = os.path.join(img_zi_dir, file)
            
            print(f"  处理图片: {file}")
            
            # 获取部首图像
            radical_images, radical_image_paths, target_radical = get_radical_images_from_img_zi(character, image_name)
            
            if not radical_image_paths:
                print(f"  跳过 {file} - 没有找到部首图像")
                continue
            
            print(f"  找到 {len(radical_image_paths)} 个部首: {target_radical}")
            
            # 使用PrototypeClassifier预测可能的部首
            predicted_radicals = []
            
            for i, radical_image in enumerate(radical_images):
                if radical_image is not None:
                    try:
                        # 使用PrototypeClassifier预测部首
                        possible_radical = get_possible_radical(radical_image, class_prototypes, train_classes, 
                                                              model, all_radical_list, std, mean)
                        if possible_radical and len(possible_radical) > 0:
                            predicted_radicals.append(possible_radical[0])  # 取第一个（相似度最高）
                        else:
                            predicted_radicals.append(target_radical[i])  # 使用目标部首作为后备
                    except Exception as e:
                        print(f"  部首预测失败: {e}")
                        predicted_radicals.append(target_radical[i])  # 使用目标部首作为后备
                else:
                    predicted_radicals.append(target_radical[i])  # 使用目标部首作为后备
            
            print(f"  预测的部首: {predicted_radicals}")
            
            # 调用LLM进行识别
            result = get_variant_prediction(image_path, radical_image_paths, predicted_radicals)
            
            print(f"  识别结果: {result}")
            
            # 解析结果
            if result and "," in result:
                # 解析top10结果
                candidates = [c.strip() for c in result.split(",")]
                # 去重，保持顺序
                unique_candidates = []
                for c in candidates:
                    if c not in unique_candidates:
                        unique_candidates.append(c)
                
                top1_result = unique_candidates[0] if unique_candidates else result
                top5_results = unique_candidates[:5] if len(unique_candidates) >= 5 else unique_candidates
                top10_results = unique_candidates[:10] if len(unique_candidates) >= 10 else unique_candidates
                
                # 检查是否有重复
                if len(candidates) != len(unique_candidates):
                    print(f"    ⚠️ 检测到重复字符，已去重: {candidates} -> {unique_candidates}")
            else:
                top1_result = result
                top5_results = [result] if result else []
                top10_results = [result] if result else []
            
            # 保存结果
            results.append({
                'character': character,
                'image_path': image_path,
                'radical_image_paths': radical_image_paths,
                'predicted_radicals': predicted_radicals,
                'llm_result': result,
                'top1_result': top1_result,
                'top5_results': top5_results,
                'top10_results': top10_results
            })
            
            cnt += 1
    
    # 5. 输出总结
    print(f"\n=== 处理完成 ===")
    print(f"总异体字数: {len(variant_chars)}")
    print(f"成功处理数: {len(results)}")
    
    print(f"\n=== 详细结果 ===")
    for result in results:
        print(f"字符: {result['character']}")
        print(f"Top1结果: {result['top1_result']}")
        print(f"Top5结果: {result['top5_results']}")
        print(f"Top10结果: {result['top10_results']}")
        print(f"完整结果: {result['llm_result']}")
        print("-" * 50)
    
    # 6. 计算准确度
    if results:
        print(f"\n📊 准确度计算:")
        
        top1_correct = 0
        top5_correct = 0
        top10_correct = 0
        total_count = len(results)
        
        for result in results:
            ground_truth = result['character']
            top1_result = result['top1_result']
            top5_results = result['top5_results']
            top10_results = result['top10_results']
            
            # 检查Top1准确度
            if top1_result == ground_truth:
                top1_correct += 1
            
            # 检查Top5准确度
            if ground_truth in top5_results:
                top5_correct += 1
            
            # 检查Top10准确度
            if ground_truth in top10_results:
                top10_correct += 1
            
            print(f"  字符: {ground_truth}")
            print(f"    Top1: {top1_result} {'✅' if top1_result == ground_truth else '❌'}")
            print(f"    Top5: {top5_results} {'✅' if ground_truth in top5_results else '❌'}")
            print(f"    Top10: {top10_results} {'✅' if ground_truth in top10_results else '❌'}")
            print(f"    完整结果: {result['llm_result']}")
            print("-" * 50)
        
        # 计算准确度
        top1_accuracy = top1_correct / total_count if total_count > 0 else 0
        top5_accuracy = top5_correct / total_count if total_count > 0 else 0
        top10_accuracy = top10_correct / total_count if total_count > 0 else 0
        
        print(f"\n🎯 最终结果:")
        print(f"总测试字符数: {total_count}")
        print(f"Top1正确数: {top1_correct}")
        print(f"Top5正确数: {top5_correct}")
        print(f"Top10正确数: {top10_correct}")
        print(f"Top1准确度: {top1_accuracy:.2%}")
        print(f"Top5准确度: {top5_accuracy:.2%}")
        print(f"Top10准确度: {top10_accuracy:.2%}")
    
    return results



def main():
    """
    主函数：处理所有异体字识别
    """
    process_all_variant_characters()

if __name__ == "__main__":
    # 直接运行完整模式
    print("🚀 开始运行异体字识别系统...")
    try:
        results = process_all_variant_characters()
        if results:
            print(f"\n✅ 异体字识别完成！处理了 {len(results)} 个字符")
        else:
            print("❌ 没有处理任何字符")
    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()