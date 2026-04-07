#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存管理器 - 为知识图谱查询提供缓存机制
"""

import json
import hashlib
import time
from typing import Dict, List, Optional, Any
from collections import OrderedDict
import os

class KnowledgeGraphCache:
    """知识图谱查询缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache", max_size: int = 1000, ttl: int = 3600):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存文件目录
            max_size: 最大缓存条目数
            ttl: 缓存生存时间（秒）
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.ttl = ttl
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 内存缓存（LRU）
        self.memory_cache = OrderedDict()
        
        # 缓存统计
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    def _generate_cache_key(self, query_type: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 创建查询的唯一标识
        query_str = f"{query_type}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, query_type: str, params: Dict[str, Any]) -> Optional[str]:
        """
        从缓存获取结果
        
        Args:
            query_type: 查询类型（如'radical_search', 'character_search'）
            params: 查询参数
            
        Returns:
            缓存的结果，如果不存在或过期则返回None
        """
        cache_key = self._generate_cache_key(query_type, params)
        
        # 1. 检查内存缓存
        if cache_key in self.memory_cache:
            result, timestamp = self.memory_cache[cache_key]
            if time.time() - timestamp < self.ttl:
                # 更新LRU顺序
                self.memory_cache.move_to_end(cache_key)
                self.stats["hits"] += 1
                return result
            else:
                # 过期，删除
                del self.memory_cache[cache_key]
        
        # 2. 检查磁盘缓存
        cache_file = self._get_cache_file_path(cache_key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # 检查是否过期
                if time.time() - cache_data["timestamp"] < self.ttl:
                    # 加载到内存缓存
                    self.memory_cache[cache_key] = (cache_data["result"], cache_data["timestamp"])
                    self.memory_cache.move_to_end(cache_key)
                    
                    # 维护缓存大小
                    if len(self.memory_cache) > self.max_size:
                        self.memory_cache.popitem(last=False)
                        self.stats["evictions"] += 1
                    
                    self.stats["hits"] += 1
                    return cache_data["result"]
                else:
                    # 过期，删除文件
                    os.remove(cache_file)
            except Exception as e:
                print(f"读取缓存文件失败: {e}")
        
        self.stats["misses"] += 1
        return None
    
    def set(self, query_type: str, params: Dict[str, Any], result: str) -> None:
        """
        设置缓存
        
        Args:
            query_type: 查询类型
            params: 查询参数
            result: 查询结果
        """
        cache_key = self._generate_cache_key(query_type, params)
        timestamp = time.time()
        
        # 1. 更新内存缓存
        self.memory_cache[cache_key] = (result, timestamp)
        self.memory_cache.move_to_end(cache_key)
        
        # 维护缓存大小
        if len(self.memory_cache) > self.max_size:
            self.memory_cache.popitem(last=False)
            self.stats["evictions"] += 1
        
        # 2. 保存到磁盘缓存
        cache_file = self._get_cache_file_path(cache_key)
        try:
            cache_data = {
                "result": result,
                "timestamp": timestamp,
                "query_type": query_type,
                "params": params
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存文件失败: {e}")
    
    def clear(self) -> None:
        """清空所有缓存"""
        # 清空内存缓存
        self.memory_cache.clear()
        
        # 清空磁盘缓存
        for file in os.listdir(self.cache_dir):
            if file.endswith('.json'):
                try:
                    os.remove(os.path.join(self.cache_dir, file))
                except Exception as e:
                    print(f"删除缓存文件失败: {e}")
        
        # 重置统计
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": f"{hit_rate:.2%}",
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_files": len([f for f in os.listdir(self.cache_dir) if f.endswith('.json')])
        }
    
    def warm_up_cache(self, common_queries: List[Dict[str, Any]]) -> None:
        """
        预热缓存
        
        Args:
            common_queries: 常见查询列表
        """
        print("🔥 开始预热缓存...")
        for query in common_queries:
            # 这里可以预加载一些常见的查询结果
            pass
        print("✅ 缓存预热完成")

# 全局缓存实例
knowledge_cache = KnowledgeGraphCache()

def cached_radical_search(radical: str) -> Optional[str]:
    """带缓存的部首搜索"""
    return knowledge_cache.get("radical_search", {"radical": radical})

def cached_character_search(character: str) -> Optional[str]:
    """带缓存的字符搜索"""
    return knowledge_cache.get("character_search", {"character": character})

def cached_radical_explanation_search(radical: str) -> Optional[str]:
    """带缓存的部首解释搜索"""
    return knowledge_cache.get("radical_explanation", {"radical": radical})

# 装饰器：为函数添加缓存功能
def cache_result(query_type: str):
    """缓存装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 构建参数字典
            params = {}
            if args:
                params["args"] = list(args)
            if kwargs:
                params.update(kwargs)
            
            # 尝试从缓存获取
            cached_result = knowledge_cache.get(query_type, params)
            if cached_result is not None:
                return cached_result
            
            # 执行原函数
            result = func(*args, **kwargs)
            
            # 缓存结果
            knowledge_cache.set(query_type, params, result)
            
            return result
        return wrapper
    return decorator
