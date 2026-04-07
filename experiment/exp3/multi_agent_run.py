#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多智能体甲骨文解释系统
模仿 run_prototype_kg.py，但只使用KG pipeline
包含两个智能体：
1. 图片分析智能体：负责分析图片、预测部首、搜索KG数据库
2. 思考总结智能体：负责对搜索信息进行深度思考和总结
"""

import sys
import os
import argparse
import random
import csv
import re
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import cv2
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier

# 添加父目录到路径，以便导入robust_csv_reader
sys.path.append('..')
from robust_csv_reader import robust_read_csv

# 导入现有模块
from KG_construct import KG_construct_new
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

# 定义工具函数供智能体调用
def search_radical_info_tool(radical: str) -> str:
    """搜索部首解释信息"""
    try:
        result = search_radical_explanation(radical)
        return result if result and "知识库中暂无" not in result else f"未找到部首'{radical}'的解释"
    except Exception as e:
        return f"搜索部首'{radical}'失败: {e}"

def search_characters_by_radical_tool(radical: str) -> str:
    """搜索包含指定部首的字符"""
    try:
        result = search_character_by_radical(radical)
        return result if result and "知识库中暂无" not in result else f"未找到包含部首'{radical}'的字符"
    except Exception as e:
        return f"搜索部首'{radical}'相关字符失败: {e}"

def search_variant_characters_tool(character: str) -> str:
    """搜索变体字符信息"""
    try:
        result = search_variant_characters(character)
        return result if result and "知识库中暂无" not in result else f"未找到字符'{character}'的变体信息"
    except Exception as e:
        return f"搜索字符'{character}'变体失败: {e}"

def search_modern_character_tool(character: str) -> str:
    """搜索现代字符对应关系"""
    try:
        result = search_character_by_modern_character(character)
        return result if result and "知识库中暂无" not in result else f"未找到字符'{character}'的现代对应关系"
    except Exception as e:
        return f"搜索字符'{character}'现代对应关系失败: {e}"

# 全局变量
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 多智能体系统的LLM配置
class MultiAgentLLMConfig:
    """多智能体系统的LLM配置类"""
    
    def __init__(self):
        # 图片分析智能体的LLM配置
        self.image_agent_model = os.getenv("IMAGE_AGENT_MODEL", "your_model_here")
        self.image_agent_base_url = os.getenv("IMAGE_AGENT_BASE_URL", "https://api.openai.com/v1")
        self.image_agent_api_key = os.getenv("IMAGE_AGENT_API_KEY", "your_api_key_here")
        self.image_agent_temperature = float(os.getenv("IMAGE_AGENT_TEMPERATURE", "0.2"))
        self.image_agent_max_tokens = int(os.getenv("IMAGE_AGENT_MAX_TOKENS", "2000"))
        
        # 思考总结智能体的LLM配置
        self.thinking_agent_model = os.getenv("THINKING_AGENT_MODEL", "your_model_here")
        self.thinking_agent_base_url = os.getenv("THINKING_AGENT_BASE_URL", "https://api.openai.com/v1")
        self.thinking_agent_api_key = os.getenv("THINKING_AGENT_API_KEY", "your_api_key_here")
        self.thinking_agent_temperature = float(os.getenv("THINKING_AGENT_TEMPERATURE", "0.3"))
        self.thinking_agent_max_tokens = int(os.getenv("THINKING_AGENT_MAX_TOKENS", "3000"))
        
        # 通用配置
        self.timeout = int(os.getenv("LLM_TIMEOUT", "120"))  # 增加超时时间到120秒
        
        # Thinking模式支持
        self.enable_thinking = os.getenv("LLM_ENABLE_THINKING", "true").lower() == "true"
        self.auto_downgrade = os.getenv("LLM_AUTO_DOWNGRADE", "true").lower() == "true"
        
        # 支持的thinking模型（通过环境变量配置）
        self.thinking_models = os.getenv("LLM_THINKING_MODELS", "").split(",") if os.getenv("LLM_THINKING_MODELS") else []
    
    def get_image_agent_llm(self, disable_tools=False):
        """获取图片分析智能体的LLM"""
        from langchain_openai import ChatOpenAI
        from langchain.tools import tool
        
        # 定义工具
        @tool
        def search_radical_info(radical: str) -> str:
            """搜索部首解释信息"""
            return search_radical_info_tool(radical)
        
        @tool
        def search_characters_by_radical(radical: str) -> str:
            """搜索包含指定部首的字符"""
            return search_characters_by_radical_tool(radical)
        
        @tool
        def search_variant_characters(character: str) -> str:
            """搜索变体字符信息"""
            return search_variant_characters_tool(character)
        
        @tool
        def search_modern_character(character: str) -> str:
            """搜索现代字符对应关系"""
            return search_modern_character_tool(character)
        
        llm_config = {
            "model": self.image_agent_model,
            "api_key": self.image_agent_api_key,
            "base_url": self.image_agent_base_url,
            "temperature": self.image_agent_temperature,
            "max_tokens": self.image_agent_max_tokens,
            "timeout": self.timeout,
        }
        
        if disable_tools:
            llm_config.update({
                "tools": [],
                "tool_choice": "none",
            })
            print(f"🔒 图片分析智能体LLM配置: 模型={self.image_agent_model}, 工具调用=禁用")
        else:
            llm_config.update({
                "tools": [search_radical_info, search_characters_by_radical, search_variant_characters, search_modern_character],
                "tool_choice": "auto",
            })
            print(f"🔧 图片分析智能体LLM配置: 模型={self.image_agent_model}, 工具调用=启用")
        
        return ChatOpenAI(**llm_config)
    
    def get_thinking_agent_llm(self, disable_tools=True):
        """获取思考总结智能体的LLM"""
        from langchain_openai import ChatOpenAI
        
        llm_config = {
            "model": self.thinking_agent_model,
            "api_key": self.thinking_agent_api_key,
            "base_url": self.thinking_agent_base_url,
            "temperature": self.thinking_agent_temperature,
            "max_tokens": self.thinking_agent_max_tokens,
            "timeout": self.timeout,
        }
        
        # 处理thinking模式
        if self.enable_thinking:
            if self.thinking_agent_model in self.thinking_models:
                print(f"✅ 思考总结智能体启用thinking模式，使用模型: {self.thinking_agent_model}")
                llm_config.update({
                    "tools": [],
                    "tool_choice": "none",
                })
                
                # 对于某些模型，可能需要额外的thinking参数
                # 根据模型名称自动适配thinking参数
                if "reasoner" in self.thinking_agent_model:
                    llm_config["thinking"] = True
                elif "r1" in self.thinking_agent_model:
                    print(f"    ℹ️ 模型内置thinking功能，无需额外参数")
                elif "instruct" in self.thinking_agent_model:
                    llm_config["reasoning"] = True
            else:
                if self.auto_downgrade:
                    print(f"⚠️ 模型 '{self.thinking_agent_model}' 不支持thinking模式，自动降级到普通模式")
                    print(f"   支持的thinking模型: {', '.join(self.thinking_models)}")
                else:
                    print(f"⚠️ 警告: 模型 '{self.thinking_agent_model}' 不支持thinking模式")
                    print(f"   支持的thinking模型: {', '.join(self.thinking_models)}")
        
        if disable_tools:
            llm_config.update({
                "tools": [],
                "tool_choice": "none",
            })
        
        print(f"🧠 思考总结智能体LLM配置: 模型={self.thinking_agent_model}, thinking={'启用' if self.enable_thinking and self.thinking_agent_model in self.thinking_models else '禁用'}")
        
        return ChatOpenAI(**llm_config)

# 全局LLM配置实例
llm_config = MultiAgentLLMConfig()

class ImageAnalysisAgent:
    """图片分析智能体：负责分析图片、预测部首、搜索KG数据库"""
    
    def __init__(self, model, class_prototypes, train_classes, std, mean, all_radical_list):
        self.model = model
        self.class_prototypes = class_prototypes
        self.train_classes = train_classes
        self.std = std
        self.mean = mean
        self.all_radical_list = all_radical_list
        
        # 创建图片分析专用的LLM（使用独立的配置）
        self.llm = llm_config.get_image_agent_llm(disable_tools=False)
        
    def search_kg_database_with_tools(self, character, best_radicals):
        """使用工具调用搜索KG数据库"""
        print(f"    🔍 图片分析智能体使用工具调用搜索KG数据库...")
        
        # 构建工具调用提示词
        kg_search_prompt = f"""
你是一个专业的甲骨文知识图谱搜索专家。请使用可用的工具搜索以下字符的相关信息：

字符: {character}
预测的部首: {best_radicals}

请按以下步骤进行搜索：
1. 首先搜索部首解释信息
2. 然后搜索包含这些部首的字符
3. 尝试搜索变体字符
4. 最后搜索现代字符对应关系

请使用工具获取信息，然后整理成结构化的搜索结果。
"""
        
        try:
            messages = [
                {"role": "user", "content": kg_search_prompt}
            ]
            
            response = self.llm.invoke(messages)
            kg_output = response.content if hasattr(response, 'content') else str(response)
            
            print(f"    📝 KG工具搜索结果: {kg_output[:100]}...")
            return kg_output
            
        except Exception as e:
            print(f"    ❌ KG工具搜索失败: {e}")
            return ""
    
    def _search_kg_database(self, character, radical_list):
        """搜索KG数据库"""
        print(f"    🔍 开始KG数据库搜索，字符: {character}, 部首: {radical_list}")
        
        # 1. 首先尝试通过部首搜索
        if radical_list:
            radical_results = []
            
            for radical in radical_list:
                print(f"    🔍 搜索部首: {radical}")
                # 搜索部首解释
                radical_explanation = search_radical_explanation(radical)
                if "知识库中暂无" not in radical_explanation and "查询失败" not in radical_explanation:
                    radical_results.append(f"部首'{radical}': {radical_explanation}")
                    print(f"    ✅ 找到部首解释: {radical}")
                
                # 搜索包含该部首的字符
                character_by_radical = search_character_by_radical(radical)
                if "知识库中暂无" not in character_by_radical and "查询失败" not in character_by_radical:
                    radical_results.append(f"包含部首'{radical}'的字符: {character_by_radical}")
                    print(f"    ✅ 找到相关字符: {radical}")
            
            if radical_results:
                combined_result = "基于部首分析:\n" + "\n".join(radical_results)
                print(f"    ✅ KG数据库搜索成功，找到 {len(radical_results)} 条信息")
                return combined_result
        
        # 2. 尝试搜索变体字符
        print(f"    🔍 尝试搜索变体字符...")
        try:
            variant_result = search_variant_characters(character)
            if "知识库中暂无" not in variant_result and "查询失败" not in variant_result:
                print(f"    ✅ 找到变体字符信息")
                return f"基于变体字符分析:\n{variant_result}"
        except Exception as e:
            print(f"    ❌ 变体字符搜索失败: {e}")
        
        # 3. 尝试通过现代字符搜索
        print(f"    🔍 尝试搜索现代字符...")
        try:
            modern_result = search_character_by_modern_character(character)
            if "知识库中暂无" not in modern_result and "查询失败" not in modern_result:
                print(f"    ✅ 找到现代字符信息")
                return f"基于现代字符分析:\n{modern_result}"
        except Exception as e:
            print(f"    ❌ 现代字符搜索失败: {e}")
        
        print(f"    ⚠️ KG数据库搜索未找到有效信息")
        return ""

    def analyze_radicals_with_llm(self, character, best_radicals):
        """使用图片分析智能体的LLM分析部首预测结果"""
        print(f"    🧠 图片分析智能体LLM分析部首预测...")
        
        radical_analysis_prompt = f"""
你是一个专业的甲骨文部首分析专家。请分析以下部首预测结果，判断哪些部首最可能正确，并给出分析理由。

字符: {character}
预测的部首: {best_radicals}

请分析：
1. 哪些部首预测最可能正确？
2. 基于甲骨文字形特征，这些部首的合理性如何？
3. 建议的搜索优先级顺序

请简洁回答，格式：
最可能部首: [部首列表]
分析理由: [简要说明]
搜索建议: [优先级排序]
"""
        
        try:
            messages = [
                {"role": "user", "content": radical_analysis_prompt}
            ]
            
            response = self.llm.invoke(messages)
            analysis_output = response.content if hasattr(response, 'content') else str(response)
            
            print(f"    📝 部首分析结果: {analysis_output[:100]}...")
            return analysis_output
            
        except Exception as e:
            print(f"    ❌ 图片分析智能体LLM调用失败: {e}")
            return ""

    def enhance_kg_search_with_llm(self, character, kg_output):
        """使用图片分析智能体的LLM增强KG检索结果"""
        print(f"    🧠 图片分析智能体LLM增强KG检索结果...")
        
        kg_enhancement_prompt = f"""
你是一个专业的甲骨文知识图谱分析专家。请基于以下KG检索结果，提取最相关的信息并重新组织。

字符: {character}
KG检索结果:
{kg_output}

请：
1. 提取与字符释义最相关的信息
2. 去除冗余和无关信息
3. 按重要性重新排序
4. 补充可能的缺失信息

请直接输出增强后的KG信息：
"""
        
        try:
            messages = [
                {"role": "user", "content": kg_enhancement_prompt}
            ]
            
            response = self.llm.invoke(messages)
            enhanced_output = response.content if hasattr(response, 'content') else str(response)
            
            print(f"    📝 KG增强结果: {enhanced_output[:100]}...")
            return enhanced_output
            
        except Exception as e:
            print(f"    ❌ 图片分析智能体LLM增强失败: {e}")
            return kg_output  # 如果失败，返回原始KG结果

    def generate_summary_with_thinking_agent(self, thinking_agent, character, best_radicals):
        """由图片分析智能体统一编排：使用工具调用搜索KG，然后调用思考智能体生成释义"""
        # 1) 图片分析智能体LLM分析部首预测
        radical_analysis = self.analyze_radicals_with_llm(character, best_radicals)
        
        # 2) 使用工具调用搜索KG数据库
        kg_output = self.search_kg_database_with_tools(character, best_radicals)
        
        # 3) 图片分析智能体LLM增强KG检索结果
        enhanced_kg_output = self.enhance_kg_search_with_llm(character, kg_output)
        
        # 4) 调用思考总结智能体
        final_output = thinking_agent.think_and_summarize(character, enhanced_kg_output)
        return final_output

class ThinkingAgent:
    """思考总结智能体：负责对搜索信息进行深度思考和总结"""
    
    def __init__(self):
        # 创建思考专用的LLM（使用独立的配置）
        self.llm = llm_config.get_thinking_agent_llm(disable_tools=True)
    
    def think_and_summarize(self, character, kg_output):
        """基于KG信息进行深度思考和总结，生成释义"""
        print(f"    🧠 思考总结智能体开始工作...")
        
        thinking_prompt = f"""
你是一个专业的甲骨文学者。请基于以下KG数据库信息，对这个甲骨文字符进行深度思考，给出简洁的释义。

字符: {character}

KG数据库信息:
{kg_output}

请直接输出这个甲骨文字符的释义，格式要求：
- 先给出简洁释义
- 然后分析部首构成和象形特征
- 不要包含冗长的文化背景或历史描述

示例格式：
报告。從口，象用口發聲之形。

请直接给出释义：
"""
        
        # 增加重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"    🔄 思考总结智能体处理中... (尝试 {attempt + 1}/{max_retries})")
                
                # 使用thinking模式进行深度思考
                messages = [
                    {"role": "user", "content": thinking_prompt}
                ]
                
                response = self.llm.invoke(messages)
                full_output = response.content if hasattr(response, 'content') else str(response)
                
                # 检查输出是否为空
                if full_output and full_output.strip():
                    # 打印完整的思考过程
                    print(f"    📝 思考过程:")
                    print(f"    {'='*50}")
                    print(f"    {full_output}")
                    print(f"    {'='*50}")
                    
                    # 提取简洁释义（只保留第一句话或前100个字符）
                    clean_output = self._extract_concise_meaning(full_output)
                    
                    print(f"    ✅ 思考总结智能体完成思考")
                    print(f"    📄 简洁释义: {clean_output}")
                    return clean_output
                else:
                    print(f"    ⚠️ 第 {attempt + 1} 次尝试返回空内容，重试...")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(3)  # 等待3秒后重试
                        continue
                    else:
                        print(f"    ❌ 所有重试都失败，返回空内容")
                        return ""
                
            except Exception as e:
                print(f"    ❌ 思考总结智能体第 {attempt + 1} 次尝试失败: {e}")
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    print(f"    ⏰ 检测到超时错误，等待更长时间后重试...")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(5)  # 超时后等待更长时间
                        continue
                elif attempt < max_retries - 1:
                    import time
                    time.sleep(3)  # 其他错误等待3秒后重试
                    continue
                else:
                    print(f"    ❌ 所有重试都失败，返回空内容")
                    return ""
    
    def _extract_concise_meaning(self, output_text):
        """从完整输出中提取最简洁的释义，只保留核心释义内容"""
        if not output_text or output_text.strip() == "":
            return ""
        
        cleaned_output = output_text.strip()
        
        # 移除可能的格式标记
        cleaned_output = re.sub(r'^[-\*•\s]+', '', cleaned_output)
        cleaned_output = re.sub(r'[-\*•\s]+$', '', cleaned_output)
        
        # 移除思考过程相关的关键词
        thinking_keywords = [
            "让我分析", "根据", "从图片", "从数据库", "综合分析", "识别关键", 
            "推理", "考虑", "验证", "思考过程", "分析结果", "数据库信息",
            "图片特征", "部首构成", "关联分析", "象形含义", "历史背景",
            "文化内涵", "关键线索", "证据", "可能含义", "用途", "合理性",
            "在商代", "甲骨文占卜", "祭祀儀式", "超自然力量", "神權", "敬畏",
            "掌控意圖", "先民", "信仰", "反映了", "體現了", "常用於", "向神灵",
            "祖先報告", "事件", "通過", "與", "溝通的", "對", "的", "和"
        ]
        
        # 查找标准释义格式：字符,释义。從...，象...之形。
        pattern = r'([^，。！？]*?)([，。！？])(.*?從.*?象.*?之形[。！？]?)'
        match = re.search(pattern, cleaned_output, re.DOTALL)
        
        if match:
            # 找到标准格式的释义
            prefix = match.group(1).strip()
            separator = match.group(2)
            description = match.group(3).strip()
            
            # 只保留核心释义，去掉冗长的文化背景描述
            # 提取到"之形"为止，去掉后面的文化背景
            core_pattern = r'(.*?從.*?象.*?之形)'
            core_match = re.search(core_pattern, description)
            if core_match:
                core_description = core_match.group(1).strip()
                if prefix and core_description:
                    return f"{prefix}{separator}{core_description}。"
            else:
                # 如果没有找到标准格式，只取到第一个句号
                first_sentence = description.split('。')[0] + '。'
                if prefix and first_sentence:
                    return f"{prefix}{separator}{first_sentence}"
        
        # 如果没有找到标准格式，尝试提取第一句简洁释义
        sentences = re.split(r'[。！？]', cleaned_output)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 3:  # 至少3个字符
                # 检查是否包含思考过程关键词
                is_thinking = any(keyword in sentence for keyword in thinking_keywords)
                if not is_thinking:
                    return sentence + "。"
        
        # 如果都包含思考关键词，取第一句
        if sentences and sentences[0].strip():
            return sentences[0].strip() + "。"
        
        # 最后兜底，取前50个字符
        if len(cleaned_output) > 50:
            return cleaned_output[:50] + "..."
        
        return cleaned_output
    
    def _clean_output(self, output_text):
        """清理输出文本（保留用于兼容性）"""
        return self._extract_concise_meaning(output_text)

def split_data_for_kg_and_test(csv_file, train_ratio=0.7, random_seed=42):
    """将数据分割为训练集（用于构建KG）和测试集（用于测试LLM）"""
    print(f"📊 开始数据分割...")
    print(f"   训练集比例: {train_ratio:.1%}")
    print(f"   测试集比例: {1-train_ratio:.1%}")
    print(f"   随机种子: {random_seed}")
    
    # 读取数据
    df = robust_read_csv(csv_file, expected_columns=2)
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
    """使用训练集数据构建知识图谱"""
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
    base_data_dir = Path(__file__).resolve().parents[1] / 'data'
    img_zi_dir = str(base_data_dir / 'img_zi' / character)
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

def process_test_characters_multi_agent(test_df, force_restart=False):
    """使用多智能体系统处理测试集字符"""
    
    # 获得PrototypeClassifier模型
    print('----------获得PrototypeClassifier模型-----------')
    model, class_prototypes, train_classes, std, mean = get_prototype_model()
    if model is None:
        print("❌ 无法获取PrototypeClassifier模型，退出")
        return
    
    model.eval()
    
    # 获取所有部首列表
    try:
        base_data_dir = Path(__file__).resolve().parents[1] / 'data'
        radical_df_all = robust_read_csv(str(base_data_dir / 'radical_explanation.csv'), expected_columns=4)
    except Exception as e:
        print(f"❌ 读取部首数据失败: {e}")
        return
    
    all_radical_list = radical_df_all['Radical'].unique().tolist()
    
    # 创建智能体
    print('🤖 初始化多智能体系统...')
    image_agent = ImageAnalysisAgent(model, class_prototypes, train_classes, std, mean, all_radical_list)
    thinking_agent = ThinkingAgent()
    print('✅ 多智能体系统初始化完成')
    
    # 使用测试集数据
    All_zi = test_df
    print(f"成功读取测试集数据: {len(All_zi)} 个字符")
    
    # 创建输出目录
    output_dir = 'output_multi_agent'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取两个智能体的模型型号用于文件名
    # 多智能体系统的CSV文件路径
    multi_agent_file = f'{output_dir}/test_set_multi_agent.csv'
    
    # 检查是否强制重新开始
    if force_restart:
        if os.path.exists(multi_agent_file):
            os.remove(multi_agent_file)
            print(f"🗑️ 强制重新开始，删除现有结果文件: {multi_agent_file}")
        print("🆕 将从头开始处理所有字符")
    
    # 检查是否已有结果文件，支持断点续传
    processed_characters = set()
    start_index = 0
    
    if os.path.exists(multi_agent_file):
        try:
            df = pd.read_csv(multi_agent_file)
            if len(df) > 0:
                # 过滤掉表头行，只统计实际数据
                data_rows = df[df['Character'] != 'Character']  # 排除表头
                if len(data_rows) > 0:
                    processed_chars = set(data_rows['Character'].tolist())
                    processed_characters.update(processed_chars)
                    print(f"✅ 找到已处理的结果文件: {multi_agent_file}，包含 {len(processed_chars)} 个字符")
                else:
                    print(f"⚠️ 文件存在但只有表头: {multi_agent_file}")
        except Exception as e:
            print(f"⚠️ 读取文件失败: {multi_agent_file}, 错误: {e}")
    
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
        print(f"📁 CSV文件路径: {multi_agent_file}")
        # 创建CSV文件并写入表头
        with open(multi_agent_file, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(["Character", "Ground_Truth", "LLM_Output", "Pipeline"])
        print(f"✅ CSV文件已创建: {multi_agent_file}")
    
    # 读取ground truth数据
    base_data_dir = Path(__file__).resolve().parents[1] / 'data'
    gt_df = robust_read_csv(str(base_data_dir / 'character_explanations_CN.csv'), expected_columns=2)
    
    # 处理每个字符
    all_characters = All_zi['Character'].tolist()
    
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
            
            # 多智能体协作处理（由图片分析智能体统一编排调用思考智能体）
            print(f"  🤖 多智能体协作处理开始...")
            final_output = image_agent.generate_summary_with_thinking_agent(thinking_agent, zi, best_radicals)
            
            # 保存结果
            print(f"  💾 保存结果到CSV: {zi} -> {final_output[:50]}...")
            with open(multi_agent_file, mode='a', newline='', encoding='utf-8-sig', errors='ignore') as file:
                writer = csv.writer(file)
                writer.writerow([zi, ground_truth, final_output, "Multi-Agent"])
            print(f"  ✅ 结果已保存到CSV")
            
            # 更新已处理字符集合，用于断点续传
            processed_characters.add(zi)
            
            # 显示进度
            remaining = len(all_characters) - len(processed_characters)
            print(f"  ✅ 字符 {zi} 处理完成，剩余 {remaining} 个字符")
            
            cnt += 1

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='多智能体甲骨文解释系统')
    parser.add_argument('--shuffle', action='store_true', default=True, 
                       help='是否打乱甲骨字顺序 (默认: True)')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                       help='不打乱甲骨字顺序')
    # 图片分析智能体LLM配置
    parser.add_argument('--image-agent-model', type=str, default=None, help='图片分析智能体LLM模型名')
    parser.add_argument('--image-agent-base-url', type=str, default=None, help='图片分析智能体LLM API 基础URL')
    parser.add_argument('--image-agent-api-key', type=str, default=None, help='图片分析智能体LLM API 密钥')
    parser.add_argument('--image-agent-temperature', type=float, default=None, help='图片分析智能体LLM 温度参数')
    parser.add_argument('--image-agent-max-tokens', type=int, default=None, help='图片分析智能体LLM 最大token数')
    
    # 思考总结智能体LLM配置
    parser.add_argument('--thinking-agent-model', type=str, default=None, help='思考总结智能体LLM模型名')
    parser.add_argument('--thinking-agent-base-url', type=str, default=None, help='思考总结智能体LLM API 基础URL')
    parser.add_argument('--thinking-agent-api-key', type=str, default=None, help='思考总结智能体LLM API 密钥')
    parser.add_argument('--thinking-agent-temperature', type=float, default=None, help='思考总结智能体LLM 温度参数')
    parser.add_argument('--thinking-agent-max-tokens', type=int, default=None, help='思考总结智能体LLM 最大token数')
    
    # 通用LLM配置
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
    
    # 运行期覆盖环境变量，供多智能体系统使用
    # 图片分析智能体配置
    if args.image_agent_model:
        os.environ["IMAGE_AGENT_MODEL"] = args.image_agent_model
    if args.image_agent_base_url:
        os.environ["IMAGE_AGENT_BASE_URL"] = args.image_agent_base_url
    if args.image_agent_api_key:
        os.environ["IMAGE_AGENT_API_KEY"] = args.image_agent_api_key
    if args.image_agent_temperature:
        os.environ["IMAGE_AGENT_TEMPERATURE"] = str(args.image_agent_temperature)
    if args.image_agent_max_tokens:
        os.environ["IMAGE_AGENT_MAX_TOKENS"] = str(args.image_agent_max_tokens)
    
    # 思考总结智能体配置
    if args.thinking_agent_model:
        os.environ["THINKING_AGENT_MODEL"] = args.thinking_agent_model
    if args.thinking_agent_base_url:
        os.environ["THINKING_AGENT_BASE_URL"] = args.thinking_agent_base_url
    if args.thinking_agent_api_key:
        os.environ["THINKING_AGENT_API_KEY"] = args.thinking_agent_api_key
    if args.thinking_agent_temperature:
        os.environ["THINKING_AGENT_TEMPERATURE"] = str(args.thinking_agent_temperature)
    if args.thinking_agent_max_tokens:
        os.environ["THINKING_AGENT_MAX_TOKENS"] = str(args.thinking_agent_max_tokens)
    
    # 通用配置
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
    img_zi_root = str((Path(__file__).resolve().parents[1] / 'data' / 'img_zi'))
    if not os.path.exists(img_zi_root):
        print("data/img_zi 目录不存在")
        exit(1)
    
    # 处理测试集字符
    print('🚀 开始多智能体处理测试集字符...')
    process_test_characters_multi_agent(test_df, force_restart=args.force_restart)
    
    print(f'\n🎉 所有字符处理完成!')
    
    # 输出缓存统计
    print('\n📊 缓存统计信息:')
    cache_stats = get_cache_stats()
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")
    
    # 输出结果统计
    output_dir = 'output_multi_agent'
    # 多智能体系统的结果文件
    multi_agent_file = f'{output_dir}/test_set_multi_agent.csv'
    
    print("🎉 多智能体系统的结果已保存到output_multi_agent目录")
    print(f"📊 训练集字符数: {len(train_df)}")
    print(f"📊 测试集字符数: {len(test_df)}")
    
    # 统计结果数量
    if os.path.exists(multi_agent_file):
        df = pd.read_csv(multi_agent_file)
        print(f"📊 多智能体系统总样本数: {len(df)}")
    else:
        print(f"❌ 多智能体系统结果文件不存在: {multi_agent_file}")
    
    print(f"\n📁 输出文件:")
    print(f"  多智能体系统: {multi_agent_file}")
