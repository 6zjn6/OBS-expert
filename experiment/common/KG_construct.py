from py2neo import Graph, Node, Relationship
import pandas as pd
import re
import os

neo_auth_key = "87654321"

def KG_construct_new():
    """
    构建汉字知识图谱
    使用data文件夹中的CSV文件：
    - character_explanations_CN.csv: 包含汉字及其解释
    - radical_explanation.csv: 包含部首及其解释
    """
    
    # 连接Neo4j数据库
    graph = Graph("bolt://localhost:7687", auth=("neo4j", neo_auth_key))
    graph.delete_all()  # 清空数据库
    
    # 读取CSV文件
    try:
        # 从环境变量获取字符数据文件路径，如果没有则使用默认路径
        character_csv_file = os.environ.get('CHARACTER_CSV_FILE', 'data/character_explanations_CN.csv')
        df_characters = pd.read_csv(character_csv_file)
        
        # 读取部首数据，添加错误处理
        try:
            df_radicals = pd.read_csv('data/radical_explanation.csv')
        except pd.errors.ParserError as e:
            print(f"⚠️ 部首CSV解析错误: {e}")
            print("🔧 尝试使用错误处理...")
            try:
                # 尝试使用更宽松的参数
                df_radicals = pd.read_csv('data/radical_explanation.csv', 
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
                with open('data/radical_explanation.csv', 'r', encoding='utf-8-sig', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader)  # 读取标题行
                    for i, row in enumerate(reader, 1):
                        if len(row) == 4:  # 确保有4个字段
                            rows.append(row)
                        else:
                            print(f"⚠️ 跳过格式错误的行 {i}: {row}")
                df_radicals = pd.DataFrame(rows, columns=header)
        
        print(f"✅ 成功读取数据文件")
        print(f"   字符数据文件: {character_csv_file}")
        print(f"   字符数据: {len(df_characters)} 行")
        print(f"   部首数据: {len(df_radicals)} 行")
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("请确保以下文件存在:")
        print(f"  - {character_csv_file}")
        print("  - data/radical_explanation.csv")
        return
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return

    # 创建 character 节点
    print("\n📝 开始创建字符节点...")
    character_count = 0
    for i, row in df_characters.iterrows():
        if i % 100 == 0:
            print(f"   处理进度: {i}/{len(df_characters)}")
            
        character = str(row['Character']).strip()
        explanation = str(row['Explanation']).strip()
        
        if not character or not explanation:
            continue
            
        # 创建或合并 character 节点
        node_char = Node(
            'character',
            character=character,
            explanation=explanation,
            type='汉字'
        )
        graph.merge(node_char, 'character', 'character')
        character_count += 1
    
    print(f"✅ 成功创建 {character_count} 个字符节点")

    # 创建 radical 节点及其与 character 的关系
    print("\n📝 开始创建部首节点和关系...")
    radical_count = 0
    relation_count = 0
    
    # 按部首分组处理
    radical_groups = df_radicals.groupby('Radical')
    
    for radical_name, group in radical_groups:
        if radical_count % 50 == 0:
            print(f"   处理进度: {radical_count}/{len(radical_groups)}")
            
        # 清理部首名称
        radical_name = str(radical_name).strip()
        radical_name = re.sub(r"[\（\(].*?[\）\)]", "", radical_name).strip()
        radical_name = re.sub(r"\s+", "", radical_name)
        
        if not radical_name:
            continue
            
        # 收集该部首的所有解释
        explanations = []
        for _, row in group.iterrows():
            exp = str(row['Explanation']).strip()
            if exp and exp not in explanations:
                explanations.append(exp)
        
        # 创建或合并 radical 节点
        node_rad = Node(
            'radical', 
            radical_name=radical_name,
            explanation='; '.join(explanations),
            type='部首'
        )
        graph.merge(node_rad, 'radical', 'radical_name')
        radical_count += 1
        
        # 创建与字符的关系
        for _, row in group.iterrows():
            part_of_char = str(row['Part_of_Character']).strip()
            if part_of_char:
                # 查找对应的字符节点
                node_char = graph.nodes.match('character', character=part_of_char).first()
                if node_char:
                    # 创建关系
                    rel = Relationship(node_rad, "PART_OF_CHARACTER", node_char)
                    graph.merge(rel)
                    relation_count += 1

    print(f"✅ 成功创建 {radical_count} 个部首节点")
    print(f"✅ 成功创建 {relation_count} 个关系")
    
    # 统计信息
    total_nodes = len(graph.nodes)
    total_relationships = len(graph.relationships)
    
    print(f"\n🎉 知识图谱构建完成!")
    print(f"   总节点数: {total_nodes}")
    print(f"   总关系数: {total_relationships}")
    print(f"   字符节点: {character_count}")
    print(f"   部首节点: {radical_count}")

def get_csv_format_info():
    """
    返回CSV文件所需的格式信息
    """
    print("📋 CSV文件格式要求:")
    print("\n1. character_explanations_CN.csv 格式:")
    print("   Character,Explanation")
    print("   一,数字1，最小的正整数；也表示单一、整体或开始")
    print("   丁,天干第四位；古代指城邑；也用作人名或时间标记")
    print("   ...")
    
    print("\n2. radical_explanation.csv 格式:")
    print("   Radical,File_Path,Part_of_Character,Explanation")
    print("   一,0-一-甲-前4.47.61_一.png,一,数字1或最小的整数")
    print("   丁,379-丁-甲-合集3096_丁.png,丁,跟方形或城邑有关的东西")
    print("   ...")
    
    print("\n字段说明:")
    print("  - Character: 汉字字符")
    print("  - Explanation: 字符的解释说明")
    print("  - Radical: 部首名称")
    print("  - File_Path: 图片文件路径（可选）")
    print("  - Part_of_Character: 该部首属于哪个汉字")
    print("  - Explanation: 部首的解释说明")

if __name__ == "__main__":
    # 检查文件是否存在
    if not os.path.exists('data/character_explanations_CN.csv'):
        print("❌ 缺少文件: data/character_explanations_CN.csv")
        get_csv_format_info()
    elif not os.path.exists('data/radical_explanation.csv'):
        print("❌ 缺少文件: data/radical_explanation.csv")
        get_csv_format_info()
    else:
        KG_construct_new()




