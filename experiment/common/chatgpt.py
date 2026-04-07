import os.path
from os import system
from types import new_class
# from openai import OpenAI
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
import base64
from langchain.prompts import FewShotPromptTemplate, PromptTemplate,ChatPromptTemplate,MessagesPlaceholder,FewShotChatMessagePromptTemplate
from langchain_core.messages import (
    HumanMessage,SystemMessage
)
import requests

# from tests.eval import cal_CIDEr
from PIL import Image
import io
from langchain_core.output_parsers import StrOutputParser
# Function to encode the image
def encode_image(image_path,output_size=(128,128),quality=85):
  # with open(image_path, "rb") as image_file:
  #   return base64.b64encode(image_file.read()).decode('utf-8')
    image = Image.open(image_path)
    # 调整图像大小,降低一下图片大小
    image = image.resize(output_size, Image.Resampling.LANCZOS)

    # 将图像保存到内存中，并指定压缩质量
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=quality)
    image_data=img_byte_arr.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    return base64_image  # 返回压缩后的二进制图像数据

neokey = os.getenv("NEO4J_PASSWORD", "87654321")
here_API = os.getenv("LLM_API_KEY", "your_api_key_here")

def chat_with_gpt_new_noimage(image_path,radical_list): # 不输入图像
    import os
    # # web:chatanywhere
    model = ChatOpenAI(model=os.getenv("LLM_MODEL", "your_model_here"), api_key=here_API,
                       base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),temperature=0.2)

    # 构建图像编码
    base64_image = encode_image(image_path)
    # 构建知识增强
    # -------------------连接图数据库
    from py2neo import Graph, Node, Relationship, NodeMatcher
    import pandas as pd
    import numpy as np
    import re
    character_knowledge=[]
    radical_knowledge=[]
    for radical in radical_list:
        # 连接Neo4j数据库
        graph = Graph('bolt://localhost:7687', auth=("neo4j", neokey))

        ans = "" # 获得包含部首的字的解释
        ans2="" # 获得单一部首的解释
        # 枚举这一部分的可能部首
        for i in radical:
            # 构造查询
            query = """
                       MATCH (n:character)
                        WHERE n.radical_list contains $radical or n.mordernc contains $radical
                       RETURN n.both as clue
                    """

            result = graph.run(query,radical=i)

            # 执行查询并获取结果
            for idx,record in enumerate(result):
                ans+=(record["clue"])+';'+'\n'
                if idx>=2:
                    break
            query= """
                       MATCH (n:radical)
                        WHERE n.radical_name contains $radical
                       RETURN n.explanation as clue
                    """
            result = graph.run(query, radical=i)

            # 执行查询并获取结果
            for idx, record in enumerate(result):
                ans2 += (record["clue"]) + ';' + '\n'

        character_knowledge.append(ans)
        radical_knowledge.append(ans2)
    # print(knowledge)
    # exit(0)
    clue=""
    for index,(i,j) in enumerate(zip(character_knowledge,radical_knowledge)):
        clue+=f"The part {index+1}'s possible radicals are {radical_list[index]}."
        clue+="The displayed list is sorted by the likelihood of radicals from highest to lowest),and the explanation of radicals and characters containing these radicals is as follows. \n"
        clue+="Radical:\n"
        clue+=j+'\n'
        clue+="Character:\n"
        clue+=i+'\n'
    print(clue)
    # exit(0)
    # 修改后的格式
    examples = [
        {
            "input": "Here is an image of an oracle bone script...",
            "output":
                "- Represents the sun setting among grass and trees, indicating dusk or evening.\n"
                "- Represents the human form with an emphasized head, originally meaning 'head' and extended to signify the first position or beginning.\n"
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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Assume you are an oracle bone script expert from China,describe the image provided"),
            few_shot_prompt,
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": (
                            "Based on the given prompt and the structure of the character, select the five most appropriate combinations from the provided list of possible radicals and infer their reasonable explanations that you think are the most likely."
                            "Clue：{clue}"
                            "\nPlease refer to the explanations of different components in various characters from the hints, and based on the provided list of possible components, provide a comprehensive interpretation of this oracle bone script character"
                            "\nNote that the prompt may not be entirely accurate. Please combine the oracle bone script images and your understanding to respond."
                            "Strictly follow the example's format(Each answer starts with '-' and use a newline between each answer).No more symbol.Do not include any extra symbols or line breaks.Only provide the explanation part, do not mention the radical composition.\n"
                            "No matter what, you need to provide the five answers you believe is the most reasonable."
                        )

                    },

                ],
            ),

        ]
    )
    # print(prompt)
    chain = (prompt | model)


    response = chain.invoke({"clue": clue})
    # print(response.content)
    return response.content  # 新的clue构架

def chat_with_gpt_new_bothimage(image_path,radical_image_paths,radical_list): #输入字符和部首两个级别的图像
    import os
    # # web:chatanywhere
    model = ChatOpenAI(model=os.getenv("LLM_MODEL", "your_model_here"), api_key=here_API,
                       base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),temperature=0.2)

    # 构建图像编码
    base64_image = encode_image(image_path)
    base64_radical=[]
    for ele in radical_image_paths:
        base64_radical.append(encode_image(ele))
    # 构建知识增强
    # -------------------连接图数据库
    from py2neo import Graph, Node, Relationship, NodeMatcher
    import pandas as pd
    import numpy as np
    import re
    character_knowledge=[]
    radical_knowledge=[]
    for radical in radical_list:
        # 连接Neo4j数据库
        graph = Graph('bolt://localhost:7687', auth=("neo4j", neokey))

        ans = "" # 获得包含部首的字的解释
        ans2="" # 获得单一部首的解释
        # 枚举这一部分的可能部首
        for i in radical:
            # 构造查询
            query = """
                       MATCH (n:character)
                        WHERE n.radical_list contains $radical or n.mordernc contains $radical
                       RETURN n.both as clue
                    """

            result = graph.run(query,radical=i)

            # 执行查询并获取结果
            for idx,record in enumerate(result):
                ans+=(record["clue"])+';'+'\n'
                if idx>=2:
                    break
            query= """
                       MATCH (n:radical)
                        WHERE n.radical_name contains $radical
                       RETURN n.explanation as clue
                    """
            result = graph.run(query, radical=i)

            # 执行查询并获取结果
            for idx, record in enumerate(result):
                ans2 += (record["clue"]) + ';' + '\n'

        character_knowledge.append(ans)
        radical_knowledge.append(ans2)
    # print(knowledge)
    # exit(0)
    clue=""
    for index,(i,j) in enumerate(zip(character_knowledge,radical_knowledge)):
        clue+=f"The part {index+1}'s possible radicals are {radical_list[index]}."
        clue+="The displayed list is sorted by the likelihood of radicals from highest to lowest),and the explanation of radicals and characters containing these radicals is as follows. \n"
        clue+="Radical:\n"
        clue+=j+'\n'
        clue+="Character:\n"
        clue+=i+'\n'
    print(clue)
    # exit(0)
    examples = [

        {
            "input": "Here is an image of an oracle bone script. Based on the given prompt and the structure of the character, infer five reasonable explanations that you think are the most likely.The image is as follows.",
            "output":
                "- Represents the sun setting among grass and trees, indicating dusk or evening.\n"
                "- Represents the human form with an emphasized head, originally meaning 'head' and extended to signify the first position or beginning.\n"
                "- The two dots symbolize a woman's breasts, representing a nurturing mother.\n"
                "- The original meaning is believed to be dispute or contention.\n"
                "- The character depicts a hand capturing a bird, representing the act of capturing or hunting.\n"
        },
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

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", "Assume you are an oracle bone script expert from China,describe the image provided"),
    #         few_shot_prompt,
    #         (
    #             "user",
    #             [
    #                 {
    #                     "type": "text",
    #                     "text": (
    #                         "Here is an image of an oracle bone script. Based on the given prompt and the structure of the character, select the five most appropriate combinations from the provided list of possible radicals and infer their reasonable explanations that you think are the most likely.The image is as follows."
    #                         "Clue：{clue}"
    #                         "\nPlease refer to the explanations of different components in various characters from the hints, and based on the provided list of possible components, provide a comprehensive interpretation of this oracle bone script character"
    #                         "\nNote that the prompt may not be entirely accurate. Please combine the oracle bone script images and your understanding to respond."
    #                         "Strictly follow the example's format(Each answer starts with '-' and use a newline between each answer).No more symbol.Do not include any extra symbols or line breaks.Include two parts,'The oracle bone...','Explanation'.\n"
    #                         "No matter what, you need to provide the five answers you believe is the most reasonable."
    #                     )
    #
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {"url": "data:image/jpeg;base64,{base64_image}"},
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {"url": "data:image/jpeg;base64,{base64_radical1}"},
    #                 },
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {"url": "data:image/jpeg;base64,{base64_radical2}"},
    #                 }
    #
    #             ],
    #         ),
    #
    #     ]
    # )
    # print(prompt)


    if len(base64_radical)==0:
        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", "Assume you are an oracle bone script expert from China,describe the image provided"),
        #         few_shot_prompt,
        #         (
        #             "user",
        #             [
        #                 {
        #                     "type": "text",
        #                     "text": (
        #                         "Here is an image of an oracle bone script. Based on the given prompt and the structure of the character, select the five most appropriate combinations from the provided list of possible radicals and infer their reasonable explanations that you think are the most likely.The image is as follows."
        #                         "Clue：{clue}"
        #                         "\nPlease refer to the explanations of different components in various characters from the hints, and based on the provided list of possible components, provide a comprehensive interpretation of this oracle bone script character"
        #                         "\nNote that the prompt may not be entirely accurate. Please combine the oracle bone script images and your understanding to respond."
        #                         "Strictly follow the example's format(Each answer starts with '-' and use a newline between each answer).No more symbol.Do not include any extra symbols or line breaks.Include two parts,'The oracle bone...','Explanation'.\n"
        #                         "No matter what, you need to provide the five answers you believe is the most reasonable."
        #                     )
        #
        #                 },
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {"url": "data:image/jpeg;base64,{base64_image}"},
        #                 }
        #             ],
        #         ),
        #
        #     ]
        # )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 """## Role Definition
        You are a multimodal oracle bone script analysis expert trained by archaeologists and AI engineers, with capabilities to:
        1. Analyze both visual and semantic features
        2. Understand topological evolution patterns of radicals
        3. Detect conflicts between model predictions and knowledge graph

        ## Workflow
        1. Image Feature Extraction: Analyze stroke patterns and structural composition
        2. Radical Candidate Validation: Compare with MoCo model's radical exemplars ({base64_radical1})
        3. Semantic Network Reasoning: Integrate knowledge graph relationships"""),

                few_shot_prompt,  # Maintain original examples

                ("user", [
                    {
                        "type": "text",
                        "text":
                            """## Input Data

        ## Analysis Requirements
        1. Quintuple Selection Strategy:
           - Must select 5 combinations from candidates
           - Top 3 confidence radicals must be included
           - Each combination ≤3 radicals

        2. Conflict Filtering:
           ! Mark radicals conflicting with visual structure with <UNCERTAIN>

        3. Output Specifications:
           - Start each answer with "- " (English dash)
           - Only provide the semantic explanation
           - Do not mention radical composition"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}
                    },
                ])
            ]
        )
        chain = (prompt | model)
        response = chain.invoke({"base64_image": base64_image, "clue": clue})
        # print("GPT Response:", response, '\n')
    elif len(base64_radical)==1:
        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", "Assume you are an oracle bone script expert from China,describe the image provided"),
        #         few_shot_prompt,
        #         (
        #             "user",
        #             [
        #                 {
        #                     "type": "text",
        #                     "text": (
        #                         "Here is an image of an oracle bone script. Based on the given prompt and the structure of the character, select the five most appropriate combinations from the provided list of possible radicals and infer their reasonable explanations that you think are the most likely.The image is as follows."
        #                         "Clue：{clue}"
        #                         "\nPlease refer to the explanations of different components in various characters from the hints, and based on the provided list of possible components, provide a comprehensive interpretation of this oracle bone script character"
        #                         "\nNote that the prompt may not be entirely accurate. Please combine the oracle bone script images and your understanding to respond."
        #                         "Strictly follow the example's format(Each answer starts with '-' and use a newline between each answer).No more symbol.Do not include any extra symbols or line breaks.Include two parts,'The oracle bone...','Explanation'.\n"
        #                         "No matter what, you need to provide the five answers you believe is the most reasonable."
        #                     )
        #
        #                 },
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {"url": "data:image/jpeg;base64,{base64_image}"},
        #                 },
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {"url": "data:image/jpeg;base64,{base64_radical1}"},
        #                 }
        #             ],
        #         ),
        #
        #     ]
        # )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 """## Role Definition
        You are a multimodal oracle bone script analysis expert trained by archaeologists and AI engineers, with capabilities to:
        1. Analyze both visual and semantic features
        2. Understand topological evolution patterns of radicals
        3. Detect conflicts between model predictions and knowledge graph
        
        ## Workflow
        1. Image Feature Extraction: Analyze stroke patterns and structural composition
        2. Radical Candidate Validation: Compare with MoCo model's radical exemplars ({base64_radical1})
        3. Semantic Network Reasoning: Integrate knowledge graph relationships"""),

                few_shot_prompt,  # Maintain original examples

                ("user", [
                    {
                        "type": "text",
                        "text":
                            """## Input Data

        ## Analysis Requirements
        1. Quintuple Selection Strategy:
           - Must select 5 combinations from candidates
           - Top 3 confidence radicals must be included
           - Each combination ≤3 radicals

        2. Conflict Filtering:
           ! Mark radicals conflicting with visual structure with <UNCERTAIN>

        3. Output Specifications:
           - Start each answer with "- " (English dash)
           - Only provide the semantic explanation
           - Do not mention radical composition"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{base64_radical1}"}
                    }
                ])
            ]
        )
        chain = (prompt | model)
        response = chain.invoke({"base64_image": base64_image, "clue": clue,"base64_radical1":base64_radical[0]})
    else :

        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", "Assume you are an oracle bone script expert from China,describe the image provided"),
        #         few_shot_prompt,
        #         (
        #             "user",
        #             [
        #                 {
        #                     "type": "text",
        #                     "text": (
        #                         "Here is an image of an oracle bone script. Based on the given prompt and the structure of the character, select the five most appropriate combinations from the provided list of possible radicals and infer their reasonable explanations that you think are the most likely.The image is as follows."
        #                         "Clue：{clue}"
        #                         "\nPlease refer to the explanations of different components in various characters from the hints, and based on the provided list of possible components, provide a comprehensive interpretation of this oracle bone script character"
        #                         "\nNote that the prompt may not be entirely accurate. Please combine the oracle bone script images and your understanding to respond."
        #                         "Strictly follow the example's format(Each answer starts with '-' and use a newline between each answer).No more symbol.Do not include any extra symbols or line breaks.Include two parts,'The oracle bone...','Explanation'.\n"
        #                         "No matter what, you need to provide the five answers you believe is the most reasonable."
        #                     )
        #
        #                 },
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {"url": "data:image/jpeg;base64,{base64_image}"},
        #                 },
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {"url": "data:image/jpeg;base64,{base64_radical1}"},
        #                 },
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {"url": "data:image/jpeg;base64,{base64_radical2}"},
        #                 }
        #             ],
        #         ),
        #
        #     ]
        # )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 """## Role Definition
        You are a multimodal oracle bone script analysis expert trained by archaeologists and AI engineers, with capabilities to:
        1. Analyze both visual and semantic features
        2. Understand topological evolution patterns of radicals
        3. Detect conflicts between model predictions and knowledge graph

        ## Workflow
        1. Image Feature Extraction: Analyze stroke patterns and structural composition
        2. Radical Candidate Validation: Compare with MoCo model's radical exemplars ({base64_radical1})
        3. Semantic Network Reasoning: Integrate knowledge graph relationships"""),

                few_shot_prompt,  # Maintain original examples

                ("user", [
                    {
                        "type": "text",
                        "text":
                            """## Input Data

        ## Analysis Requirements
        1. Quintuple Selection Strategy:
           - Must select 5 combinations from candidates
           - Top 3 confidence radicals must be included
           - Each combination ≤3 radicals


        2. Output Specifications:
           - Start each answer with "- " (English dash)
           - Only provide the semantic explanation
           - Do not mention radical composition"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{base64_radical1}"}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{base64_radical2}"},
                    }
                ])
            ]
        )
        chain = (prompt | model)
        response = chain.invoke({"base64_image": base64_image, "clue": clue,"base64_radical1":base64_radical[0],"base64_radical2":base64_radical[1]})





    # print(response.content)
    print("GPT Response:", response.content, '\n')
    return response.content  # 新的clue构架



def chat_with_gpt_new_noimage_english(image_path,radical_list): # 不输入图像，英文输出格式
    import os
    # # web:chatanywhere
    model = ChatOpenAI(model=os.getenv("LLM_MODEL", "your_model_here"), api_key=here_API,
                       base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),temperature=0.2)

    # 构建图像编码
    base64_image = encode_image(image_path)
    # 构建知识增强
    # -------------------连接图数据库
    from py2neo import Graph, Node, Relationship, NodeMatcher
    import pandas as pd
    import numpy as np
    import re
    character_knowledge=[]
    radical_knowledge=[]
    for radical in radical_list:
        # 连接Neo4j数据库
        graph = Graph('bolt://localhost:7687', auth=("neo4j", neokey))

        ans = "" # 获得包含部首的字的解释
        ans2="" # 获得单一部首的解释
        # 枚举这一部分的可能部首
        for i in radical:
            # 构造查询
            query = """
                       MATCH (n:character)
                        WHERE n.radical_list contains $radical or n.mordernc contains $radical
                       RETURN n.both as clue
                    """

            result = graph.run(query,radical=i)

            # 执行查询并获取结果
            for idx,record in enumerate(result):
                ans+=(record["clue"])+';'+'\n'
                if idx>=2:
                    break
            query= """
                       MATCH (n:radical)
                        WHERE n.radical_name contains $radical
                       RETURN n.explanation as clue
                    """
            result = graph.run(query, radical=i)

            # 执行查询并获取结果
            for idx, record in enumerate(result):
                ans2 += (record["clue"]) + ';' + '\n'

        character_knowledge.append(ans)
        radical_knowledge.append(ans2)
    # print(knowledge)
    # exit(0)
    clue=""
    for index,(i,j) in enumerate(zip(character_knowledge,radical_knowledge)):
        clue+=f"The part {index+1}'s possible radicals are {radical_list[index]}."
        clue+="The displayed list is sorted by the likelihood of radicals from highest to lowest),and the explanation of radicals and characters containing these radicals is as follows. \n"
        clue+="Radical:\n"
        clue+=j+'\n'
        clue+="Character:\n"
        clue+=i+'\n'
    print(clue)
    # exit(0)
    examples = [
        {
            "input": "Based on the given prompt and the structure of the character, infer five reasonable explanations that you think are the most likely.",
            "output":
                "- The character consists of two components, both identified as 人, which typically represents a person or people. This suggests a meaning related to human activity or social interaction.\n"
                "- The character is composed of 日 (sun) and 茻 (grass/trees), representing the sun setting among vegetation, indicating dusk or evening time.\n"
                "- The character contains 女 (woman) and two dots, where the dots symbolize a woman's breasts, representing nurturing and motherhood.\n"
                "- The character consists of 又 (hand) and 隹 (bird), depicting a hand capturing a bird, representing the act of hunting or capturing.\n"
                "- The character is formed by 女 (woman) and 女 (woman), suggesting a meaning related to dispute or contention between women."
        },
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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Assume you are an oracle bone script expert from China,describe the image provided"),
            few_shot_prompt,
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": (
                            "Based on the given prompt and the structure of the character, select the five most appropriate combinations from the provided list of possible radicals and infer their reasonable explanations that you think are the most likely."
                            "Clue：{clue}"
                            "\nPlease refer to the explanations of different components in various characters from the hints, and based on the provided list of possible components, provide a comprehensive interpretation of this oracle bone script character"
                            "\nNote that the prompt may not be entirely accurate. Please combine the oracle bone script images and your understanding to respond."
                            "Strictly follow the example's format(Each answer starts with '-' and use a newline between each answer).No more symbol.Do not include any extra symbols or line breaks.Only provide the explanation part, do not mention the radical composition.\n"
                            "No matter what, you need to provide the five answers you believe is the most reasonable."
                        )

                    },

                ],
            ),

        ]
    )
    # print(prompt)
    chain = (prompt | model)


    response = chain.invoke({"clue": clue})
    # print(response.content)
    return response.content  # 新的clue构架

def chat_with_gpt_new_bothimage_english(image_path,radical_image_paths,radical_list): #输入字符和部首两个级别的图像，英文输出格式
    import os
    # # web:chatanywhere
    model = ChatOpenAI(model=os.getenv("LLM_MODEL", "your_model_here"), api_key=here_API,
                       base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),temperature=0.2)

    # 构建图像编码
    base64_image = encode_image(image_path)
    base64_radical=[]
    for ele in radical_image_paths:
        base64_radical.append(encode_image(ele))
    # 构建知识增强
    # -------------------连接图数据库
    from py2neo import Graph, Node, Relationship, NodeMatcher
    import pandas as pd
    import numpy as np
    import re
    character_knowledge=[]
    radical_knowledge=[]
    for radical in radical_list:
        # 连接Neo4j数据库
        graph = Graph('bolt://localhost:7687', auth=("neo4j", neokey))

        ans = "" # 获得包含部首的字的解释
        ans2="" # 获得单一部首的解释
        # 枚举这一部分的可能部首
        for i in radical:
            # 构造查询
            query = """
                       MATCH (n:character)
                        WHERE n.radical_list contains $radical or n.mordernc contains $radical
                       RETURN n.both as clue
                    """

            result = graph.run(query,radical=i)

            # 执行查询并获取结果
            for idx,record in enumerate(result):
                ans+=(record["clue"])+';'+'\n'
                if idx>=2:
                    break
            query= """
                       MATCH (n:radical)
                        WHERE n.radical_name contains $radical
                       RETURN n.explanation as clue
                    """
            result = graph.run(query, radical=i)

            # 执行查询并获取结果
            for idx, record in enumerate(result):
                ans2 += (record["clue"]) + ';' + '\n'

        character_knowledge.append(ans)
        radical_knowledge.append(ans2)
    # print(knowledge)
    # exit(0)
    clue=""
    for index,(i,j) in enumerate(zip(character_knowledge,radical_knowledge)):
        clue+=f"The part {index+1}'s possible radicals are {radical_list[index]}."
        clue+="The displayed list is sorted by the likelihood of radicals from highest to lowest),and the explanation of radicals and characters containing these radicals is as follows. \n"
        clue+="Radical:\n"
        clue+=j+'\n'
        clue+="Character:\n"
        clue+=i+'\n'
    print(clue)
    # exit(0)
    examples = [
        {
            "input": "Here is an image of an oracle bone script. Based on the given prompt and the structure of the character, infer five reasonable explanations that you think are the most likely.The image is as follows.",
            "output":
                "Explanation: The character consists of two components, both identified as 人, which typically represents a person or people. This suggests a meaning related to human activity or social interaction.\n"
                "Explanation: The character is composed of 日 (sun) and 茻 (grass/trees), representing the sun setting among vegetation, indicating dusk or evening time.\n"
                "Explanation: The character contains 女 (woman) and two dots, where the dots symbolize a woman's breasts, representing nurturing and motherhood.\n"
                "Explanation: The character consists of 又 (hand) and 隹 (bird), depicting a hand capturing a bird, representing the act of hunting or capturing.\n"
                "Explanation: The character is formed by 女 (woman) and 女 (woman), suggesting a meaning related to dispute or contention between women."
        },
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

    if len(base64_radical)==0:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Assume you are an oracle bone script expert from China,describe the image provided"),
                few_shot_prompt,
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": (
                                "Here is an image of an oracle bone script. Based on the given prompt and the structure of the character, select the five most appropriate combinations from the provided list of possible radicals and infer their reasonable explanations that you think are the most likely.The image is as follows."
                                "Clue：{clue}"
                                "\nPlease refer to the explanations of different components in various characters from the hints, and based on the provided list of possible components, provide a comprehensive interpretation of this oracle bone script character"
                                "\nNote that the prompt may not be entirely accurate. Please combine the oracle bone script images and your understanding to respond."
                                "Strictly follow the example's format(Each answer starts with '-' and use a newline between each answer).No more symbol.Do not include any extra symbols or line breaks.Only provide the explanation part, do not mention the radical composition.\n"
                                "No matter what, you need to provide the five answers you believe is the most reasonable."
                            )

                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,{base64_image}"},
                        }
                    ],
                ),

            ]
        )
        chain = (prompt | model)
        response = chain.invoke({"base64_image": base64_image, "clue": clue})
    elif len(base64_radical)==1:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Assume you are an oracle bone script expert from China,describe the image provided"),
                few_shot_prompt,
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": (
                                "Here is an image of an oracle bone script. Based on the given prompt and the structure of the character, select the five most appropriate combinations from the provided list of possible radicals and infer their reasonable explanations that you think are the most likely.The image is as follows."
                                "Clue：{clue}"
                                "\nPlease refer to the explanations of different components in various characters from the hints, and based on the provided list of possible components, provide a comprehensive interpretation of this oracle bone script character"
                                "\nNote that the prompt may not be entirely accurate. Please combine the oracle bone script images and your understanding to respond."
                                "Strictly follow the example's format(Each answer starts with '-' and use a newline between each answer).No more symbol.Do not include any extra symbols or line breaks.Only provide the explanation part, do not mention the radical composition.\n"
                                "No matter what, you need to provide the five answers you believe is the most reasonable."
                            )

                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,{base64_image}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,{base64_radical1}"},
                        }

                    ],
                ),

            ]
        )
        chain = (prompt | model)
        response = chain.invoke({"base64_image": base64_image, "base64_radical1": base64_radical[0], "clue": clue})
    else:
        # 处理多个部首图像的情况
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Assume you are an oracle bone script expert from China,describe the image provided"),
                few_shot_prompt,
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": (
                                "Here is an image of an oracle bone script. Based on the given prompt and the structure of the character, select the five most appropriate combinations from the provided list of possible radicals and infer their reasonable explanations that you think are the most likely.The image is as follows."
                                "Clue：{clue}"
                                "\nPlease refer to the explanations of different components in various characters from the hints, and based on the provided list of possible components, provide a comprehensive interpretation of this oracle bone script character"
                                "\nNote that the prompt may not be entirely accurate. Please combine the oracle bone script images and your understanding to respond."
                                "Strictly follow the example's format(Each answer starts with '-' and use a newline between each answer).No more symbol.Do not include any extra symbols or line breaks.Only provide the explanation part, do not mention the radical composition.\n"
                                "No matter what, you need to provide the five answers you believe is the most reasonable."
                            )

                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,{base64_image}"},
                        }
                    ],
                ),

            ]
        )
        chain = (prompt | model)
        response = chain.invoke({"base64_image": base64_image, "clue": clue})

    return response.content  # 新的clue构架