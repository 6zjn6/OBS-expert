import os.path
from os import system
from types import new_class
from langchain_openai import ChatOpenAI
import base64
from langchain.prompts import FewShotPromptTemplate, PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.messages import (
    HumanMessage, SystemMessage
)
import requests
from PIL import Image
import io
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from py2neo import Graph, Node, Relationship, NodeMatcher
import pandas as pd
import numpy as np
import re

neokey = os.getenv("NEO4J_PASSWORD", "87654321")
here_API = os.getenv("LLM_API_KEY", "your_api_key_here")
# 统一的LLM构造函数，支持通过环境变量切换模型与提供商
def get_llm(disable_tools=False):
    model_name = os.getenv("LLM_MODEL", "your_model_here")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("LLM_API_KEY", here_API)
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    timeout = int(os.getenv("LLM_TIMEOUT", "60"))
    
    # 新增：thinking 模式支持
    enable_thinking = os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true"
    # 新增：自动降级选项（当模型不支持 thinking 时是否自动禁用）
    auto_downgrade = os.getenv("LLM_AUTO_DOWNGRADE", "true").lower() == "true"
    
    # 基础配置
    llm_config = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    
    # 如果禁用工具调用（用于Baseline Pipeline）
    if disable_tools:
        llm_config.update({
            "tools": [],  # 禁用工具调用
            "tool_choice": "none",  # 禁用工具调用
        })
        print(f"🔒 LLM 配置: 模型={model_name}, 工具调用=禁用")
    
    # 为支持 thinking 的模型添加特殊配置
    thinking_models = os.getenv("LLM_THINKING_MODELS", "").split(",") if os.getenv("LLM_THINKING_MODELS") else []
    
    # 处理 thinking 模式
    if enable_thinking:
        if model_name in thinking_models:
            # 模型支持 thinking 模式
            print(f"✅ 启用 thinking 模式，使用模型: {model_name}")
            
            # 添加 thinking 模式相关参数
            llm_config.update({
                "tools": [],  # thinking 模式通常不需要外部工具
                "tool_choice": "none",  # 禁用工具调用
            })
            
            # 对于某些模型，可能需要额外的 thinking 参数
            # 根据模型名称自动适配thinking参数
            if "reasoner" in model_name:
                llm_config["thinking"] = True
            elif "r1" in model_name:
                print(f"    ℹ️ 模型内置thinking功能，无需额外参数")
            elif "instruct" in model_name:
                llm_config["reasoning"] = True
        else:
            # 模型不支持 thinking 模式
            if auto_downgrade:
                # 自动降级到非 thinking 模式
                print(f"⚠️  模型 '{model_name}' 不支持 thinking 模式，自动降级到普通模式")
                print(f"   支持的 thinking 模型: {', '.join(thinking_models)}")
                print(f"   如需强制使用 thinking 模式，请设置 LLM_AUTO_DOWNGRADE=false")
                # 重置 thinking 标志
                enable_thinking = False
            else:
                # 保持用户选择，但给出警告
                print(f"⚠️  警告: 模型 '{model_name}' 不支持 thinking 模式")
                print(f"   支持的 thinking 模型: {', '.join(thinking_models)}")
                print(f"   将以普通模式运行模型: {model_name}")
                print(f"   如需自动降级，请设置 LLM_AUTO_DOWNGRADE=true")
    
    # 记录最终配置
    if enable_thinking and model_name in thinking_models:
        print(f"🔧 LLM 配置: 模型={model_name}, thinking=启用, 工具=禁用")
    else:
        print(f"🔧 LLM 配置: 模型={model_name}, thinking=禁用")
    
    return ChatOpenAI(**llm_config)



# 尝试导入高级缓存，如果失败则使用基础缓存
try:
    from rag_cache_advanced import advanced_cache, advanced_cache_result
    print("✅ 使用高级RagCache（语义相似度+向量数据库）")
    USE_ADVANCED_CACHE = True
except ImportError:
    from cache_manager import knowledge_cache, cache_result
    print("⚠️  使用基础缓存模式")
    USE_ADVANCED_CACHE = False

# Function to encode the image
def encode_image(image_path, output_size=(256, 256), quality=95):
    image = Image.open(image_path)
    # 调整图像大小,提高图片质量
    image = image.resize(output_size, Image.Resampling.LANCZOS)

    # 将图像保存到内存中，并指定压缩质量
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=quality)
    image_data = img_byte_arr.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    return base64_image  # 返回压缩后的二进制图像数据

# 安全的图像编码函数，确保不泄露任何文件名信息
def encode_image_safely(image_path, output_size=(256, 256), quality=95):
    """
    安全地编码图像，确保不泄露任何文件名或路径信息
    用于Baseline Pipeline，防止数据泄露
    """
    try:
        # 打开图像
        image = Image.open(image_path)
        
        # 移除所有可能的元数据信息
        # 创建一个新的图像对象，不包含任何元数据
        clean_image = Image.new('RGB', image.size)
        clean_image.paste(image)
        
        # 调整图像大小
        clean_image = clean_image.resize(output_size, Image.Resampling.LANCZOS)

        # 将图像保存到内存中，不包含任何元数据
        img_byte_arr = io.BytesIO()
        clean_image.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
        image_data = img_byte_arr.getvalue()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        return base64_image
    except Exception as e:
        print(f"⚠️ 安全图像编码失败: {e}")
        # 如果安全编码失败，回退到普通编码
        return encode_image(image_path, output_size, quality)



# 定义检索工具函数
@tool
def search_character_by_radical(radical: str) -> str:
    """
    根据部首名称搜索包含该部首的字符信息（带缓存）
    
    Args:
        radical: 部首名称，如 '人', '日', '女' 等
        
    Returns:
        包含该部首的字符解释信息
    """
    # 缓存装饰器
    if USE_ADVANCED_CACHE:
        func = advanced_cache_result("radical_search")(lambda r: _search_character_by_radical_impl(r))
        return func(radical)
    else:
        func = cache_result("radical_search")(lambda r: _search_character_by_radical_impl(r))
        return func(radical)

def _search_character_by_radical_impl(radical: str) -> str:
    """实际的部首搜索实现"""
    try:
        # 连接Neo4j数据库
        graph = Graph('bolt://localhost:7687', auth=("neo4j", neokey))
        
        # 尝试多种查询模式 - 修复后的查询
        queries = [
            # 模式1: 通过关系查询
            f"MATCH (r:radical)-[:PART_OF_CHARACTER]->(c:character) WHERE r.radical_name = '{radical}' RETURN c.explanation as clue LIMIT 3",
            # 模式2: 部首名称包含
            f"MATCH (r:radical)-[:PART_OF_CHARACTER]->(c:character) WHERE r.radical_name contains '{radical}' RETURN c.explanation as clue LIMIT 3",
            # 模式3: 直接字符匹配
            f"MATCH (c:character) WHERE c.character = '{radical}' RETURN c.explanation as clue LIMIT 3",
            # 模式4: 字符包含
            f"MATCH (c:character) WHERE c.character contains '{radical}' RETURN c.explanation as clue LIMIT 3"
        ]
        
        for query in queries:
            try:
                result = graph.run(query)
                character_info = ""
                for idx, record in enumerate(result):
                    if record['clue']:
                        character_info += f"{record['clue']};\n"
                    if idx >= 2:
                        break
                
                if character_info:
                    return character_info
                    
            except Exception as e:
                continue
        
        return f"知识库中暂无部首 '{radical}' 的相关信息，请基于视觉特征进行分析"
        
    except Exception as e:
        return f"查询失败: {str(e)}"

@tool
def search_radical_explanation(radical: str) -> str:
    """
    根据部首名称搜索部首的解释信息（带缓存）
    
    Args:
        radical: 部首名称，如 '人', '日', '女' 等
        
    Returns:
        部首的解释信息
    """
    # 缓存装饰器
    if USE_ADVANCED_CACHE:
        func = advanced_cache_result("radical_explanation")(lambda r: _search_radical_explanation_impl(r))
        return func(radical)
    else:
        func = cache_result("radical_explanation")(lambda r: _search_radical_explanation_impl(r))
        return func(radical)

def _search_radical_explanation_impl(radical: str) -> str:
    """实际的部首解释搜索实现"""
    try:
        # 连接Neo4j数据库
        graph = Graph('bolt://localhost:7687', auth=("neo4j", neokey))
        
        # 尝试多种查询模式
        queries = [
            # 模式1: 直接包含
            f"MATCH (n:radical) WHERE n.radical_name contains '{radical}' RETURN n.explanation as clue LIMIT 2",
            # 模式2: 精确匹配
            f"MATCH (n:radical) WHERE n.radical_name = '{radical}' RETURN n.explanation as clue LIMIT 2",
            # 模式3: 模糊匹配
            f"MATCH (n:radical) WHERE n.radical_name CONTAINS '{radical}' RETURN n.explanation as clue LIMIT 2"
        ]
        
        for query in queries:
            try:
                result = graph.run(query)
                radical_info = ""
                for idx, record in enumerate(result):
                    if record['clue']:
                        radical_info += f"{record['clue']};\n"
                    if idx >= 1:
                        break
                
                if radical_info:
                    return radical_info
                    
            except Exception as e:
                continue
        
        return f"知识库中暂无部首 '{radical}' 的解释信息，请基于视觉特征进行分析"
        
    except Exception as e:
        return f"查询失败: {str(e)}"

@tool
def search_character_by_modern_character(modern_char: str) -> str:
    """
    根据现代汉字搜索相关的甲骨文信息（带缓存）
    
    Args:
        modern_char: 现代汉字，如 '人', '日', '女' 等
        
    Returns:
        相关的甲骨文字符信息
    """
    # 缓存装饰器
    if USE_ADVANCED_CACHE:
        func = advanced_cache_result("modern_character_search")(lambda r: _search_character_by_modern_character_impl(r))
        return func(modern_char)
    else:
        func = cache_result("modern_character_search")(lambda r: _search_character_by_modern_character_impl(r))
        return func(modern_char)

@tool
def search_exact_character(character: str) -> str:
    """
    精确搜索字符的解释信息（带缓存）
    
    Args:
        character: 要搜索的字符，如 '耋', '皇', '乙' 等
        
    Returns:
        字符的准确解释信息，如果找到的话
    """
    # 缓存装饰器
    if USE_ADVANCED_CACHE:
        func = advanced_cache_result("exact_character_search")(lambda r: _search_exact_character_impl(r))
        return func(character)
    else:
        func = cache_result("exact_character_search")(lambda r: _search_exact_character_impl(r))
        return func(character)

@tool
def search_variant_characters(character: str) -> str:
    """
    搜索异体字信息（带缓存）
    
    Args:
        character: 要搜索的字符，如 '女', '学' 等
        
    Returns:
        相关的异体字信息，包括甲骨文变体
    """
    # 缓存装饰器
    if USE_ADVANCED_CACHE:
        func = advanced_cache_result("variant_character_search")(lambda r: _search_variant_characters_impl(r))
        return func(character)
    else:
        func = cache_result("variant_character_search")(lambda r: _search_variant_characters_impl(r))
        return func(character)

def _search_character_by_modern_character_impl(modern_char: str) -> str:
    """实际的现代字符搜索实现"""
    try:
        # 连接Neo4j数据库
        graph = Graph('bolt://localhost:7687', auth=("neo4j", neokey))
        
        # 尝试多种查询模式 - 修复后的查询
        queries = [
            # 模式1: 直接字符匹配
            f"MATCH (c:character) WHERE c.character = '{modern_char}' RETURN c.explanation as clue LIMIT 3",
            # 模式2: 字符包含
            f"MATCH (c:character) WHERE c.character contains '{modern_char}' RETURN c.explanation as clue LIMIT 3",
            # 模式3: 通过部首关系查询
            f"MATCH (r:radical)-[:PART_OF_CHARACTER]->(c:character) WHERE r.radical_name = '{modern_char}' RETURN c.explanation as clue LIMIT 3"
        ]
        
        for query in queries:
            try:
                result = graph.run(query)
                char_info = ""
                for idx, record in enumerate(result):
                    if record['clue']:
                        char_info += f"{record['clue']};\n"
                    if idx >= 2:
                        break
                
                if char_info:
                    return char_info
                    
            except Exception as e:
                continue
        
        return f"知识库中暂无现代汉字 '{modern_char}' 相关的甲骨文信息，请基于视觉特征进行分析"
        
    except Exception as e:
        return f"查询失败: {str(e)}"

def _search_exact_character_impl(character: str) -> str:
    """精确搜索字符的实现"""
    try:
        # 连接Neo4j数据库
        graph = Graph('bolt://localhost:7687', auth=("neo4j", neokey))
        
        # 精确匹配查询
        query = f"MATCH (c:character) WHERE c.character = '{character}' RETURN c.explanation as clue LIMIT 1"
        
        try:
            result = graph.run(query)
            for record in result:
                if record['clue']:
                    return f"找到字符 '{character}' 的准确解释: {record['clue']}"
            return f"知识库中暂无字符 '{character}' 的准确解释，请基于视觉特征进行分析"
        except Exception as e:
            return f"查询字符 '{character}' 失败: {str(e)}"
        
    except Exception as e:
        return f"连接数据库失败: {str(e)}"

def _search_variant_characters_impl(character: str) -> str:
    """搜索异体字信息的实现"""
    try:
        # 连接Neo4j数据库
        graph = Graph('bolt://localhost:7687', auth=("neo4j", neokey))
        
        # 尝试多种查询模式来找到异体字信息
        queries = [
            # 模式1: 精确匹配字符
            f"MATCH (c:character) WHERE c.character = '{character}' RETURN c.explanation as clue LIMIT 3",
            # 模式2: 包含该字符的部首关系
            f"MATCH (r:radical)-[:PART_OF_CHARACTER]->(c:character) WHERE r.radical_name = '{character}' RETURN c.explanation as clue LIMIT 3",
            # 模式3: 模糊匹配字符
            f"MATCH (c:character) WHERE c.character contains '{character}' RETURN c.explanation as clue LIMIT 3",
            # 模式4: 通过部首名称查找相关字符
            f"MATCH (r:radical)-[:PART_OF_CHARACTER]->(c:character) WHERE r.radical_name contains '{character}' RETURN c.explanation as clue LIMIT 3"
        ]
        
        all_results = []
        for query in queries:
            try:
                result = graph.run(query)
                for record in result:
                    if record['clue'] and record['clue'] not in all_results:
                        all_results.append(record['clue'])
                        if len(all_results) >= 5:  # 最多收集5个结果
                            break
                if len(all_results) >= 5:
                    break
            except Exception as e:
                continue
        
        if all_results:
            return f"找到字符 '{character}' 的相关信息: {'; '.join(all_results)}"
        else:
            return f"知识库中暂无字符 '{character}' 的相关信息，请基于视觉特征进行分析"
        
    except Exception as e:
        return f"查询异体字信息失败: {str(e)}"

def chat_with_gpt_rag_noimage(image_path, radical_list):
    """使用RAG模式，让LLM自己调用工具函数检索信息"""
    import os
    
    # 初始化模型（从环境变量读取配置）
    model = get_llm()

    # 构建图像编码
    base64_image = encode_image(image_path)
    
    # 定义工具列表
    tools = [search_character_by_radical, search_radical_explanation, search_character_by_modern_character, search_exact_character, search_variant_characters]
    
    # 创建示例 - 使用更多样化的示例避免重复
    examples = [
        {
            "input": "这是一张甲骨文字符的图像。根据给定的提示和字符结构，推断最合理的解释。",
            "output": "表示一个人站立并举起双臂，表示'人'或'人类'的概念。\n\n原因：基于对字符结构的视觉分析和从知识图谱检索到的信息，这个字符描绘了一个具有独特特征的人形，这些特征演化为表示人或个体的概念。"
        }
    ]
    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    # 系统提示
    system_prompt = """你是一个甲骨文专家，能够分析甲骨文字符并解释其含义。

## 分析原则（重要）：
**视觉分析为主，检索信息为辅**
- 首先仔细观察图像，识别字符的视觉特征和构形原理
- 检索信息仅用于验证和补充视觉判断，不能替代视觉分析
- 如果检索信息与视觉特征冲突，以视觉分析为准

## 工作流程：
1. **视觉分析**：仔细观察图像中的甲骨文字符结构
2. **识别部首**：识别可能的部首组合（基于视觉特征）
3. **智能检索**：使用工具检索相关信息，但保持视觉判断的独立性
4. **信息整合**：将视觉分析与检索信息进行合理整合
5. **最终判断**：基于视觉分析给出最终判断

## 可用工具：
- search_character_by_radical: 根据部首搜索相关字符信息
- search_radical_explanation: 搜索部首的解释信息  
- search_character_by_modern_character: 根据现代汉字搜索甲骨文信息
- search_exact_character: 精确搜索字符的解释信息（优先使用）
- search_variant_characters: 搜索异体字信息，包括甲骨文变体（特别重要）

## 检索策略：
- 部首列表按相似度排序，优先查询相似度最高的部首
- 优先使用search_variant_characters搜索异体字信息，这对识别甲骨文特别重要
- 如果检索结果包含"知识库中暂无"等无效信息，忽略该信息
- 如果检索信息与视觉分析不符，坚持视觉判断
- 检索信息仅用于增强信心，不能改变视觉分析的核心判断

## 输出要求：
- 请用中文回答
- 直接输出最合理的解释，不要使用 "- " 前缀
- 只提供语义解释，不要提及部首组成
- 输出简洁明了，一句话概括字符含义
- 不要输出多个解释，只输出一个最准确的解释
- 在解释后，以"原因："开头给出生成释义的依据，说明基于哪些视觉特征和检索到的知识得出此结论"""

    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("user", [
                {
                    "type": "text",
                    "text": (
                        "这是一张甲骨文字符的图像。请按照以下步骤进行分析：\n\n"
                        "1. **首先进行视觉分析**：仔细观察图像，识别字符的视觉特征和构形原理\n"
                        "2. **然后进行检索验证**：使用工具检索相关信息，但保持视觉判断的独立性\n"
                        "3. **最后整合信息**：将视觉分析与检索信息进行合理整合\n\n"
                        "可用于分析的部首（按相似度从高到低排序）：{radical_list}\n\n"
                        "重要提醒：\n"
                        "- 视觉分析是主要依据，检索信息仅作辅助验证\n"
                        "- 如果检索结果包含'知识库中暂无'等无效信息，请忽略\n"
                        "- 如果检索信息与视觉分析冲突，以视觉分析为准\n"
                        "- 优先使用search_exact_character进行精确搜索\n\n"
                        "请先进行视觉分析，再使用工具检索相关信息。"
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}
                }
            ]),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )

    # 创建agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=create_openai_tools_agent(model, tools, prompt),
        tools=tools,
        verbose=True,
        max_iterations=3  # 限制最大迭代次数
    )

    # 执行
    response = agent_executor.invoke({
        "base64_image": base64_image,
        "radical_list": radical_list
    })
    
    return response["output"]

def chat_with_gpt_rag_bothimage(image_path, radical_image_paths, radical_list):
    """使用RAG模式，输入字符和部首图像，让LLM自己调用工具函数检索信息"""
    import os
    
    # 初始化模型（从环境变量读取配置）
    model = get_llm()

    # 构建图像编码
    base64_image = encode_image(image_path)
    base64_radical = []
    for ele in radical_image_paths:
        base64_radical.append(encode_image(ele))
    
    # 定义工具列表 - 只有当提供了部首列表时才使用RAG工具
    if radical_list and len(radical_list) > 0:
        tools = [search_character_by_radical, search_radical_explanation, search_character_by_modern_character, search_exact_character, search_variant_characters]
        # Generation Module方法 - 使用RAG工具
        examples = [
            {
                "input": "分析甲骨文字符图像。",
                "output": "character_type: 象形字\nreasoning: 象枝柯之形"
            },
            {
                "input": "分析甲骨文字符图像。",
                "output": "character_type: 会意字\nreasoning: 从人从木"
            },
            {
                "input": "分析甲骨文字符图像。",
                "output": "character_type: 形声字\nreasoning: 从人可声"
            }
        ]
    else:
        tools = []  # baseline方法不使用RAG工具
        # Baseline方法 - 不使用RAG工具，纯视觉推理
        examples = [
            {
                "input": "分析甲骨文字符图像。",
                "output": "character_type: 象形字\nreasoning: 象枝柯之形"
            },
            {
                "input": "分析甲骨文字符图像。",
                "output": "character_type: 会意字\nreasoning: 从人从木"
            },
            {
                "input": "分析甲骨文字符图像。",
                "output": "character_type: 形声字\nreasoning: 从人可声"
            }
        ]
    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    # 系统提示 - 根据是否使用RAG工具调整提示
    if radical_list and len(radical_list) > 0:
        # Generation Module方法 - 使用RAG工具
        system_prompt = """你是甲骨文专家。分析图像，结合检索到的信息，必须输出结果。

## 汉字构字法基础（重要）：
**象形字**：直接描摹事物的外形，如"人"象人形，"日"象太阳形
**会意字**：由两个或多个部件组合，意义相加，如"休"从人从木，表示人靠在树上休息
**形声字**：由形旁（表意）和声旁（表音）组成，如"河"从水可声，"闻"从耳门声

## 分析策略（重要）：
1. **视觉分析**：仔细观察图像特征，识别字符的视觉结构
2. **构字原理分析**：基于汉字构字法，分析字符的构形原理
3. **信息验证**：使用检索信息验证和补充分析结果

## 形声字识别指导（特别重要）：
- **形声字特征**：通常包含明显的形旁（表意部件）和声旁（表音部件）
- **形旁识别**：形旁通常表示字的意义类别（如水、木、口、人等）
- **声旁识别**：声旁通常表示字的读音，可能独立成字
- **判断标准**：如果字符包含表意部件和表音部件的组合，优先考虑形声字
- **常见形旁**：水（氵）、木、口、人（亻）、心、手（扌）、足（⻊）、目、耳等
- **常见声旁**：可、门、乐、亘、弗、正、必、斤、鬼、射等

## 决策逻辑：
- 如果检索信息与构字原理一致 → 增强信心，给出更准确的判断
- 如果检索信息与构字原理冲突 → 以构字原理为准，忽略冲突的信息
- 如果检索信息不足或无意义 → 基于构字原理进行分析
- 如果视觉特征模糊 → 结合构字原理和检索信息进行判断

## 输出格式（必须严格遵循）：
character_type: [象形字/会意字/形声字]
reasoning: [构形原理，如"象枝柯之形"或"从人可声"]

## 要求：
- 只输出上述格式，不要其他内容
- reasoning要简洁，避免通用表达，不要提及信息来源
- 优先基于构字原理，结合视觉特征进行分析"""
    else:
        # Baseline方法 - 不使用RAG工具，纯视觉推理
        system_prompt = """你是甲骨文专家。分析图像，直接输出结果。

## 汉字构字法基础（重要）：
**象形字**：直接描摹事物的外形，如"人"象人形，"日"象太阳形
**会意字**：由两个或多个部件组合，意义相加，如"休"从人从木，表示人靠在树上休息
**形声字**：由形旁（表意）和声旁（表音）组成，如"河"从水可声，"闻"从耳门声

## 形声字识别指导：
- 形声字通常包含明显的形旁（表意部件）和声旁（表音部件）
- 形旁通常表示字的意义类别（如水、木、口、人等）
- 声旁通常表示字的读音，可能独立成字
- 如果字符包含表意部件和表音部件的组合，优先考虑形声字

## 输出格式（必须严格遵循）：
character_type: [象形字/会意字/形声字]
reasoning: [构形原理，如"象枝柯之形"或"从人可声"]

## 要求：
- 只输出上述格式，不要其他内容
- reasoning要简洁，避免通用表达，不要提及信息来源"""

    # 根据部首图像数量构建不同的提示
    if len(base64_radical) == 0:
        # 只有主字符图像
        if radical_list and len(radical_list) > 0:
            # Generation Module方法
            user_messages = [
                {
                    "type": "text",
                    "text": "请分析甲骨文字符图像。重点分析构字原理，识别形旁和声旁。"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}
                }
            ]
        else:
            # Baseline方法
            user_messages = [
                {
                    "type": "text",
                    "text": "分析甲骨文字符图像。重点分析构字原理，识别形旁和声旁。"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}
                }
            ]
    elif len(base64_radical) == 1:
        # 主字符图像 + 1个部首图像
        if radical_list and len(radical_list) > 0:
            # Generation Module方法
            user_messages = [
                {
                    "type": "text",
                    "text": "请分析甲骨文字符和部首图像。重点分析构字原理，识别形旁和声旁。"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{base64_radical1}"}
                }
            ]
        else:
            # Baseline方法
            user_messages = [
                {
                    "type": "text",
                    "text": "分析甲骨文字符和部首图像。重点分析构字原理，识别形旁和声旁。"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{base64_radical1}"}
                }
            ]
    else:
        # 主字符图像 + 多个部首图像
        if radical_list and len(radical_list) > 0:
            # Generation Module方法
            user_messages = [
                {
                    "type": "text",
                    "text": "请分析甲骨文字符和多个部首图像。重点分析构字原理，识别形旁和声旁。"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}
                }
            ]
        else:
            # Baseline方法
            user_messages = [
                {
                    "type": "text",
                    "text": "分析甲骨文字符和多个部首图像。"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}
                }
            ]
        # 添加所有部首图像
        for i, radical_img in enumerate(base64_radical):
            user_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{radical_img}"}
            })

    # 根据是否使用RAG工具选择执行方式
    if radical_list and len(radical_list) > 0:
        # Generation Module方法 - 使用预检索策略
        print("    🔍 预检索RAG信息...")
        
        # 预检索RAG信息
        rag_info = ""
        try:
            # 现在radical_list已经是扁平化的最可能部首列表
            print(f"    📋 最可能的部首列表: {radical_list}")
            
            # 尝试从部首列表中获取信息
            for radical in radical_list[:3]:  # 尝试前3个部首
                print(f"    🔍 尝试检索部首: {radical}")
                try:
                    # 优先尝试异体字搜索
                    variant_info = search_variant_characters(radical)
                    print(f"      📝 异体字搜索结果: {variant_info[:100]}...")
                    if "找到字符" in variant_info and "知识库中暂无" not in variant_info:
                        rag_info += f"异体字信息: {variant_info}\n"
                        print(f"      ✅ 找到异体字信息，停止检索")
                        break
                    
                    # 尝试精确搜索字符
                    char_info = search_exact_character(radical)
                    print(f"      📝 精确搜索结果: {char_info[:100]}...")
                    if "找到字符" in char_info:
                        rag_info += f"字符信息: {char_info}\n"
                        print(f"      ✅ 找到字符信息，停止检索")
                        break
                    
                    # 尝试部首搜索
                    radical_info = search_character_by_radical(radical)
                    print(f"      📝 部首搜索结果: {radical_info[:100]}...")
                    if "知识库中暂无" not in radical_info:
                        rag_info += f"部首信息: {radical_info}\n"
                        print(f"      ✅ 找到部首信息，停止检索")
                        break
                        
                    # 尝试部首解释
                    explanation_info = search_radical_explanation(radical)
                    print(f"      📝 部首解释结果: {explanation_info[:100]}...")
                    if "知识库中暂无" not in explanation_info:
                        rag_info += f"部首解释: {explanation_info}\n"
                        print(f"      ✅ 找到部首解释，停止检索")
                        break
                        
                    # 尝试现代汉字搜索
                    modern_info = search_character_by_modern_character(radical)
                    print(f"      📝 现代汉字搜索结果: {modern_info[:100]}...")
                    if "知识库中暂无" not in modern_info:
                        rag_info += f"现代汉字信息: {modern_info}\n"
                        print(f"      ✅ 找到现代汉字信息，停止检索")
                        break
                    
                    print(f"      ❌ 所有搜索都失败，尝试下一个部首")
                        
                except Exception as e:
                    print(f"      ⚠️  检索部首 {radical} 时出错: {e}")
                    continue
                    
        except Exception as e:
            print(f"    ⚠️  RAG检索失败: {e}")
        
        # 构建增强的提示词
        if rag_info:
            print("    ✅ 找到RAG信息，使用增强模式")
            enhanced_system_prompt = f"""你是甲骨文专家。分析图像，结合检索到的信息，直接输出结果。

## 检索到的信息：
{rag_info}

## 汉字构字法基础（重要）：
**象形字**：直接描摹事物的外形，如"人"象人形，"日"象太阳形
**会意字**：由两个或多个部件组合，意义相加，如"休"从人从木，表示人靠在树上休息
**形声字**：由形旁（表意）和声旁（表音）组成，如"河"从水可声，"闻"从耳门声

## 分析策略（重要）：
**构字原理优先原则**：
- 首先基于汉字构字法分析字符的构形原理
- 结合视觉特征进行验证
- RAG信息用于补充和验证构字分析

**形声字识别指导（特别重要）**：
- 形声字通常包含明显的形旁（表意部件）和声旁（表音部件）
- 形旁通常表示字的意义类别（如水、木、口、人等）
- 声旁通常表示字的读音，可能独立成字
- 如果字符包含表意部件和表音部件的组合，优先考虑形声字
- 常见形旁：水（氵）、木、口、人（亻）、心、手（扌）、足（⻊）、目、耳等
- 常见声旁：可、门、乐、亘、弗、正、必、斤、鬼、射等

**信息利用策略**：
- 如果RAG信息与构字原理一致 → 增强判断信心
- 如果RAG信息与构字原理不符 → 以构字原理为准，忽略冲突的RAG信息
- 如果RAG信息包含"知识库中暂无"等无效信息 → 基于构字原理进行分析
- 如果构字原理不明确 → 结合视觉特征和RAG信息进行判断

## 输出格式（必须严格遵循）：
character_type: [象形字/会意字/形声字]
reasoning: [构形原理，如"象枝柯之形"或"从人可声"]

## 要求：
- 只输出上述格式，不要其他内容
- reasoning要简洁，避免通用表达
- 优先基于构字原理，结合视觉特征和RAG信息"""
        else:
            print("    ⚠️  未找到RAG信息，使用Baseline模式")
            enhanced_system_prompt = """你是甲骨文专家。分析图像，直接输出结果。

## 汉字构字法基础（重要）：
**象形字**：直接描摹事物的外形，如"人"象人形，"日"象太阳形
**会意字**：由两个或多个部件组合，意义相加，如"休"从人从木，表示人靠在树上休息
**形声字**：由形旁（表意）和声旁（表音）组成，如"河"从水可声，"闻"从耳门声

## 形声字识别指导：
- 形声字通常包含明显的形旁（表意部件）和声旁（表音部件）
- 形旁通常表示字的意义类别（如水、木、口、人等）
- 声旁通常表示字的读音，可能独立成字
- 如果字符包含表意部件和表音部件的组合，优先考虑形声字

## 输出格式（必须严格遵循）：
character_type: [象形字/会意字/形声字]
reasoning: [构形原理，如"象枝柯之形"或"从人可声"]

## 要求：
- 只输出上述格式，不要其他内容
- reasoning要简洁，避免通用表达"""
        
        # 使用增强的提示词
        enhanced_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", enhanced_system_prompt),
                few_shot_prompt,
                ("user", user_messages)
            ]
        )
        
        enhanced_chain = (enhanced_prompt | model)
        
        # 准备参数
        params = {
            "base64_image": base64_image,
            "radical_list": radical_list
        }
        
        # 根据部首图像数量添加参数
        if len(base64_radical) >= 1:
            params["base64_radical1"] = base64_radical[0]
        if len(base64_radical) >= 2:
            params["base64_radical2"] = base64_radical[1]

        # 执行增强模式
        response = enhanced_chain.invoke(params)
        # 如果找到RAG信息，返回False（未使用Baseline回退）
        return response.content, False
    else:
        # Baseline方法 - 直接调用模型，不使用Agent
        # 创建不包含agent_scratchpad的提示模板
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                few_shot_prompt,
                ("user", user_messages)
            ]
        )
        
        chain = (prompt | model)
        
        # 准备参数
        params = {
            "base64_image": base64_image,
            "radical_list": radical_list
        }
        
        # 根据部首图像数量添加参数
        if len(base64_radical) >= 1:
            params["base64_radical1"] = base64_radical[0]
        if len(base64_radical) >= 2:
            params["base64_radical2"] = base64_radical[1]

        # 执行
        response = chain.invoke(params)
        # 如果是Baseline方法，返回True（使用了Baseline模式）
        return response.content, True

# 包装函数，保持与原有接口兼容
def chat_with_gpt_new_noimage_wrapper(image_path, radical_list):
    """包装函数，调用RAG版本的函数"""
    return chat_with_gpt_rag_noimage(image_path, radical_list)

def chat_with_gpt_new_bothimage_wrapper(image_path, radical_image_paths, radical_list):
    """包装函数，调用RAG版本的函数"""
    result = chat_with_gpt_rag_bothimage(image_path, radical_image_paths, radical_list)
    if isinstance(result, tuple):
        return result[0], result[1]  # 返回(output, used_baseline_fallback)
    else:
        # 兼容旧版本，如果没有返回元组，默认未使用Baseline回退
        return result, False

# 缓存管理函数
def get_cache_stats():
    """获取缓存统计信息"""
    if USE_ADVANCED_CACHE:
        return advanced_cache.get_stats()
    else:
        return knowledge_cache.get_stats()

def clear_cache():
    """清空缓存"""
    if USE_ADVANCED_CACHE:
        advanced_cache.clear()
        print("✅ 高级缓存已清空")
    else:
        knowledge_cache.clear()
        print("✅ 基础缓存已清空")

def warm_up_cache():
    """预热缓存 - 预加载常见查询"""
    common_radicals = ['人', '日', '月', '水', '火', '木', '金', '土', '山', '川']
    common_characters = ['一', '二', '三', '上', '下', '中', '大', '小', '天', '地']
    
    print("🔥 开始预热缓存...")
    
    # 预热部首查询
    for radical in common_radicals:
        search_radical_explanation(radical)
        search_character_by_radical(radical)
    
    # 预热字符查询
    for char in common_characters:
        search_character_by_modern_character(char)
    
    print("✅ 缓存预热完成")
    print(f"缓存统计: {get_cache_stats()}")


def chat_with_gpt_variant_explanation(image_path, radical_image_paths, radical_list, custom_prompt=None, is_baseline=False, use_safe_encoding=False):
    """专门用于生成甲骨文字符释义的LLM调用函数
    
    Args:
        image_path: 主字符图像路径
        radical_image_paths: 部首图像路径列表
        radical_list: 部首列表
        custom_prompt: 自定义提示文本，如果提供则覆盖默认提示
        is_baseline: 是否为Baseline Pipeline，如果是则使用纯视觉分析的系统提示词
        use_safe_encoding: 是否使用安全图像编码，防止文件名泄露
    """
    import os
    
    try:
        # 检查输入参数
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            return "图像文件不存在", False
        
        # 初始化模型（从环境变量读取配置）
        # 如果是Baseline Pipeline，禁用工具调用以防止数据泄露
        model = get_llm(disable_tools=is_baseline)
        if model is None:
            print("❌ LLM模型初始化失败")
            return "LLM模型初始化失败", False

        # 构建图像编码
        try:
            # 如果使用安全编码或Baseline Pipeline，使用安全编码防止文件名泄露
            if is_baseline or use_safe_encoding:
                base64_image = encode_image_safely(image_path)
                print(f"✅ 主图像安全编码成功，长度: {len(base64_image)}")
            else:
                base64_image = encode_image(image_path)
                print(f"✅ 主图像编码成功，长度: {len(base64_image)}")
        except Exception as e:
            print(f"❌ 主图像编码失败: {e}")
            return f"主图像编码失败: {e}", False
            
        base64_radical = []
        
        for radical_path in radical_image_paths:
            if os.path.exists(radical_path):
                try:
                    # 如果使用安全编码或Baseline Pipeline，使用安全编码防止文件名泄露
                    if is_baseline or use_safe_encoding:
                        radical_base64 = encode_image_safely(radical_path)
                        print(f"✅ 部首图像安全编码成功")
                    else:
                        radical_base64 = encode_image(radical_path)
                        print(f"✅ 部首图像编码成功: {radical_path}")
                    base64_radical.append(radical_base64)
                except Exception as e:
                    print(f"⚠️ 部首图像编码失败: {radical_path}, 错误: {e}")
                    continue
        
        print(f"✅ 成功编码 {len(base64_radical)} 个部首图像")
        
        # 根据是否为Baseline Pipeline选择不同的系统提示词
        if is_baseline:
            # Baseline Pipeline专用系统提示词 - 纯视觉分析，无数据库信息引导
            variant_system_prompt = """你是一个甲骨文专家，专门分析甲骨文字符的完整含义。

## 核心任务：
基于视觉分析理解并解释甲骨文字符的完整含义。

## 图像顺序说明：
- 第一个图像是主字符图像，这是你要分析的主要对象
- 后续图像是部首图像，用于帮助你理解字符的构形原理
- 请始终以第一个图像（主字符）为主要分析对象

## 分析原则：
- 象形字：直接描摹事物的外形
- 会意字：由部件组合表达意义
- 形声字：形旁表意+声旁表音
- 指事字：用符号指示位置或状态

## 分析步骤：
1. **视觉分析**：仔细观察图像特征，识别字符的视觉结构
2. **构形理解**：结合部首图像，理解字符是如何构成的
3. **语义推断**：基于构形原理，推断这个字符在古代表达什么含义
4. **综合释义**：给出字符的完整、准确的释义

## 输出要求：
- 直接输出字符的完整释义，不要加任何前缀或格式标记
- 基于视觉分析给出具体含义
- 释义应该准确反映字符在古代的语义
- 可以包含构形说明，但重点是语义解释
- 输出简洁明了，不超过150个字

## 输出格式：
直接输出释义，不要其他内容。"""
        else:
            # 其他Pipeline的系统提示词 - 包含数据库信息引导
            variant_system_prompt = """你是一个甲骨文专家，专门分析甲骨文字符的完整含义。

## 核心任务：
分析甲骨文字符图像，结合部首信息和数据库检索信息，理解并解释这个字符的完整含义。

## 信息优先级：
1. **数据库信息**：如果提供了数据库检索结果，请基于这些信息进行解释
2. **视觉分析**：结合图像特征验证和补充数据库信息
3. **部首信息**：使用部首信息理解字符的构形原理

## 图像顺序说明：
- 第一个图像是主字符图像，这是你要分析的主要对象
- 后续图像是部首图像，用于帮助你理解字符的构形原理
- 请始终以第一个图像（主字符）为主要分析对象

## 重要理解：
- 数据库信息通常是最准确的，请优先使用
- 部首信息只是帮助你理解字符的构形原理，不是要你解释部首本身
- 你需要基于所有可用信息，分析整个字符表达什么概念
- 输出应该是这个字符的完整释义，不是部首的释义

## 分析步骤：
1. **数据库信息分析**：如果提供了数据库检索结果，请仔细分析这些信息
2. **视觉验证**：结合图像特征验证数据库信息的准确性
3. **构形理解**：结合部首图像，理解字符是如何构成的
4. **语义推断**：基于数据库信息和构形原理，推断这个字符在古代表达什么含义
5. **综合释义**：给出字符的完整、准确的释义

## 分析原则：
- 象形字：直接描摹事物的外形，如"人"象人形，"日"象太阳形
- 会意字：由部件组合表达意义，如"休"从人从木，表示人靠在树上休息
- 形声字：形旁表意+声旁表音，如"河"从水可声
- 指事字：用符号指示位置或状态，如"上"、"下"

## 输出要求：
- 直接输出字符的完整释义，不要加任何前缀或格式标记
- 如果数据库信息充足，请基于数据库信息进行解释
- 释义应该准确反映字符在古代的语义
- 可以包含构形说明，但重点是语义解释
- 输出简洁明了，不超过150个字

## 示例输出：
"数字一，象一横之形，表示最小的正整数或起始。"
"城邑，象方形城邑之形，表示城市、都邑。"
"人，象人侧身站立之形，表示人类、人物。"
"水，象水流之形，表示水或与水有关的事物。"
"休，从人从木，象人靠在树上休息之形，表示休息、停止。"
"河，从水可声，形旁水表示与水有关，声旁可表示读音，指河流。"

## 数据库信息使用说明：
- 如果提供了数据库检索结果，请基于这些信息进行解释
- 结合视觉特征和部首信息，生成准确、完整的解释
- 保持简洁明了，避免冗长 """

        # 创建用户消息
        if custom_prompt:
            # 使用自定义提示
            user_text = custom_prompt
        else:
            # 使用默认提示
            user_text = f"""请分析这个甲骨文字符的完整含义。

可参考的部首信息：{radical_list}

重要提醒：
- 第一个图像是主字符图像，这是你要分析的主要对象
- 后续图像是部首图像，用于帮助你理解字符的构形原理
- 部首信息只是帮助你理解字符的构形原理
- 你需要分析整个字符表达什么概念，不是解释部首
- 基于视觉特征和构形原理，给出字符的完整释义

请直接输出字符的释义，不要加任何格式标记。"""
        
        user_messages = [
            {
                "type": "text",
                "text": user_text
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
        
        # 添加部首图像 - 确保顺序正确
        for i, radical_base64 in enumerate(base64_radical):
            if i < 2:  # 最多添加2个部首图像
                user_messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{radical_base64}"}
                })

        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", variant_system_prompt),
            ("user", user_messages)
        ])

        # 执行 - 修复：传递正确的参数
        print(f"🔄 开始调用LLM...")
        print(f"📸 主图像大小: {len(base64_image)} 字符")
        print(f"📸 部首图像数量: {len(base64_radical)}")
        print(f"📝 部首列表: {radical_list}")
        print(f"📝 消息结构: 文本 + 主图像 + {len(base64_radical)} 个部首图像")
        
        chain = (prompt | model)
        
        # 修复：传递必要的参数
        response = chain.invoke({
            "base64_image": base64_image,
            "radical_list": radical_list
        })
        
        # 检查响应
        if not response:
            print("❌ LLM返回空响应对象")
            return "LLM返回空响应对象", False
            
        if not hasattr(response, 'content'):
            print("❌ LLM响应对象没有content属性")
            print(f"响应对象类型: {type(response)}")
            print(f"响应对象内容: {response}")
            return "LLM响应对象格式错误", False
            
        if not response.content or not response.content.strip():
            print("❌ LLM返回空内容")
            return "LLM返回空内容", False
            
        print(f"✅ LLM调用成功，响应长度: {len(response.content)}")
        print(f"📝 LLM响应预览: {response.content[:100]}...")
        
        return response.content, False  # 返回(output, used_baseline_fallback)
        
    except Exception as e:
        print(f"❌ LLM调用过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return f"LLM调用异常: {e}", False

def test_llm_call():
    """测试LLM调用是否正常工作"""
    print("🧪 开始测试LLM调用...")
    
    try:
        # 测试模型初始化
        model = get_llm()
        if model is None:
            print("❌ 模型初始化失败")
            return False
        print("✅ 模型初始化成功")
        
        # 测试简单的文本调用
        from langchain.prompts import ChatPromptTemplate
        
        test_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个测试助手，请回复'测试成功'"),
            ("user", "请确认你工作正常")
        ])
        
        chain = (test_prompt | model)
        response = chain.invoke({})
        
        if response and hasattr(response, 'content') and response.content:
            print(f"✅ 文本调用测试成功，响应: {response.content}")
            return True
        else:
            print("❌ 文本调用测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 运行测试
    test_llm_call()
