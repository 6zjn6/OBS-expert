import os
import base64
import io
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
import pandas as pd
from py2neo import Graph
import random

# Cache functionality removed
print("ℹ️ Cache functionality has been removed")

def get_llm(disable_tools=False):
    """Initialize LLM model from environment variables"""
    llm_config = {
        "model": os.getenv("LLM_MODEL", "your_model_here"),
        "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("LLM_API_KEY", "your_api_key_here"),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.2")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
    }
    
    # Add thinking mode if enabled
    if os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true":
        llm_config["enable_thinking"] = True
    
    # Disable tools if requested (for baseline pipeline)
    if disable_tools:
        llm_config["tools"] = []
    
    return ChatOpenAI(**llm_config)

# Function to encode the image
def encode_image(image_path, output_size=(256, 256), quality=95):
    image = Image.open(image_path)
    # Resize image, improve image quality
    image = image.resize(output_size, Image.Resampling.LANCZOS)

    # Save image to memory with specified compression quality
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=quality)
    image_data = img_byte_arr.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')

    return base64_image

# Safe image encoding function to ensure no filename information is leaked
def encode_image_safely(image_path, output_size=(256, 256), quality=95):
    """
    Safely encode image, ensuring no filename or path information is leaked
    Used for Baseline Pipeline to prevent data leakage
    """
    try:
        # Open image
        image = Image.open(image_path)
        
        # Remove all possible metadata information
        # Create a new image object without any metadata
        clean_image = Image.new('RGB', image.size)
        clean_image.paste(image)
        
        # Resize image
        clean_image = clean_image.resize(output_size, Image.Resampling.LANCZOS)

        # Save image to memory without any metadata
        img_byte_arr = io.BytesIO()
        clean_image.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
        image_data = img_byte_arr.getvalue()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        return base64_image
    except Exception as e:
        print(f"⚠️ Safe image encoding failed: {e}")
        # If safe encoding fails, fall back to normal encoding
        return encode_image(image_path, output_size, quality)

# Define retrieval tool functions
@tool
def search_character_by_radical(radical: str) -> str:
    """
    Search for characters that contain the specified radical
    Args:
        radical: The radical name to search for
    Returns:
        A string containing information about characters with this radical
    """
    try:
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "87654321"))
        
        # Query for characters containing this radical
        query = """
        MATCH (r:radical {radical_name: $radical})-[:PART_OF_CHARACTER]->(c:character)
        RETURN c.character as character, c.explanation as explanation
        LIMIT 10
        """
        
        result = graph.run(query, radical=radical)
        characters = list(result)
        
        if characters:
            char_info = []
            for char in characters:
                char_info.append(f"Character: {char['character']}, Explanation: {char['explanation']}")
            return f"Found {len(characters)} characters with radical '{radical}':\n" + "\n".join(char_info)
        else:
            return f"No characters found with radical '{radical}' in the knowledge base"
            
    except Exception as e:
        return f"Error searching for characters with radical '{radical}': {str(e)}"

@tool
def search_radical_explanation(radical: str) -> str:
    """
    Search for explanation of a specific radical
    Args:
        radical: The radical name to search for
    Returns:
        A string containing the explanation of the radical
    """
    try:
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "87654321"))
        
        # Query for radical explanation
        query = """
        MATCH (r:radical {radical_name: $radical})
        RETURN r.explanation as explanation
        """
        
        result = graph.run(query, radical=radical)
        radical_info = result.single()
        
        if radical_info and radical_info['explanation']:
            return f"Radical '{radical}' explanation: {radical_info['explanation']}"
        else:
            return f"No explanation found for radical '{radical}' in the knowledge base"
            
    except Exception as e:
        return f"Error searching for radical '{radical}': {str(e)}"

@tool
def search_character_by_modern_character(modern_char: str) -> str:
    """
    Search for oracle bone characters related to a modern character
    Args:
        modern_char: The modern character to search for
    Returns:
        A string containing information about related oracle bone characters
    """
    try:
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "87654321"))
        
        # Query for characters similar to the modern character
        query = """
        MATCH (c:character)
        WHERE c.character CONTAINS $modern_char OR c.explanation CONTAINS $modern_char
        RETURN c.character as character, c.explanation as explanation
        LIMIT 5
        """
        
        result = graph.run(query, modern_char=modern_char)
        characters = list(result)
        
        if characters:
            char_info = []
            for char in characters:
                char_info.append(f"Character: {char['character']}, Explanation: {char['explanation']}")
            return f"Found {len(characters)} characters related to '{modern_char}':\n" + "\n".join(char_info)
        else:
            return f"No characters found related to '{modern_char}' in the knowledge base"
            
    except Exception as e:
        return f"Error searching for characters related to '{modern_char}': {str(e)}"

@tool
def search_exact_character(character: str) -> str:
    """
    Search for exact character match in the knowledge base
    Args:
        character: The exact character to search for
    Returns:
        A string containing the character information if found
    """
    try:
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "87654321"))
        
        # Query for exact character match
        query = """
        MATCH (c:character {character: $character})
        RETURN c.character as character, c.explanation as explanation
        """
        
        result = graph.run(query, character=character)
        char_info = result.single()
        
        if char_info:
            return f"Found character '{character}': {char_info['explanation']}"
        else:
            return f"Character '{character}' not found in the knowledge base"
            
    except Exception as e:
        return f"Error searching for character '{character}': {str(e)}"

@tool
def search_variant_characters(character: str) -> str:
    """
    Search for variant forms of a character
    Args:
        character: The character to find variants for
    Returns:
        A string containing variant character information
    """
    try:
        graph = Graph("bolt://localhost:7687", auth=("neo4j", "87654321"))
        
        # Query for characters with similar explanations (potential variants)
        query = """
        MATCH (c:character)
        WHERE c.character <> $character AND c.explanation CONTAINS $character
        RETURN c.character as character, c.explanation as explanation
        LIMIT 5
        """
        
        result = graph.run(query, character=character)
        variants = list(result)
        
        if variants:
            var_info = []
            for var in variants:
                var_info.append(f"Variant: {var['character']}, Explanation: {var['explanation']}")
            return f"Found {len(variants)} variant characters for '{character}':\n" + "\n".join(var_info)
        else:
            return f"No variant characters found for '{character}' in the knowledge base"
            
    except Exception as e:
        return f"Error searching for variant characters of '{character}': {str(e)}"

# Cache functions removed

def chat_with_gpt_variant_explanation_ENG(image_path, radical_image_paths, radical_list, custom_prompt=None, is_baseline=False):
    """LLM call function specifically for generating oracle bone character explanations in English
    
    Args:
        image_path: Main character image path
        radical_image_paths: List of radical image paths
        radical_list: List of radicals
        custom_prompt: Custom prompt text, if provided, overrides default prompt
        is_baseline: Whether this is Baseline Pipeline, if so, uses pure visual analysis system prompt
    """
    import os
    
    try:
        # Check input parameters
        if not os.path.exists(image_path):
            print(f"❌ Image file does not exist: {image_path}")
            return "Image file does not exist", False
        
        # Initialize model (read configuration from environment variables)
        # If Baseline Pipeline, disable tool calls to prevent data leakage
        model = get_llm(disable_tools=is_baseline)
        if model is None:
            print("❌ LLM model initialization failed")
            return "LLM model initialization failed", False

        # Build image encoding
        try:
            # If Baseline Pipeline, use safe encoding to prevent filename leakage
            if is_baseline:
                base64_image = encode_image_safely(image_path)
                print(f"✅ Main image safely encoded, length: {len(base64_image)}")
            else:
                base64_image = encode_image(image_path)
                print(f"✅ Main image encoded, length: {len(base64_image)}")
        except Exception as e:
            print(f"❌ Main image encoding failed: {e}")
            return f"Main image encoding failed: {e}", False
            
        base64_radical = []
        
        for radical_path in radical_image_paths:
            if os.path.exists(radical_path):
                try:
                    # If Baseline Pipeline, use safe encoding to prevent filename leakage
                    if is_baseline:
                        radical_base64 = encode_image_safely(radical_path)
                        print(f"✅ Radical image safely encoded")
                    else:
                        radical_base64 = encode_image(radical_path)
                        print(f"✅ Radical image encoded: {radical_path}")
                    base64_radical.append(radical_base64)
                except Exception as e:
                    print(f"⚠️ Radical image encoding failed: {radical_path}, error: {e}")
                    continue
        
        print(f"✅ Successfully encoded {len(base64_radical)} radical images")
        
        # Choose different system prompts based on whether it's Baseline Pipeline
        if is_baseline:
            # Baseline Pipeline - pure visual analysis system prompt
            variant_system_prompt = """You are an oracle bone script expert, specializing in analyzing oracle bone characters through pure visual analysis.

## Core Task:
Analyze oracle bone character images and explain the complete meaning of this character based on visual features only.

## Image Order Description:
- The first image is the main character image, which is the primary object you need to analyze
- Subsequent images are radical images to help you understand the character's construction principles
- Always focus on the first image (main character) as the primary analysis object

## Important Understanding:
- You can only rely on visual analysis, no external knowledge base information
- Radical images only help you understand the character's construction principles, not to explain the radicals themselves
- You need to analyze what concept the entire character expresses based on visual features
- Output should be the complete explanation of this character, not the explanation of radicals

## Analysis Steps:
1. **Visual Analysis**: Carefully observe the visual features of the main character image
2. **Construction Understanding**: Combine radical images to understand how the character is constructed
3. **Semantic Inference**: Based on visual features and construction principles, infer what meaning this character expressed in ancient times
4. **Comprehensive Explanation**: Give a complete and accurate explanation of the character

## Analysis Principles:
- Pictographic characters: Directly depict the shape of things, such as "人" (person) resembling human form, "日" (sun) resembling sun shape
- Ideographic characters: Express meaning through component combination, such as "休" (rest) from person + tree, indicating a person resting against a tree
- Phonetic characters: Meaning component + sound component, such as "河" (river) from water + sound
- Indicative characters: Use symbols to indicate position or state, such as "上" (up), "下" (down)

## Output Requirements:
- Directly output the complete explanation of the character, without any prefix or format markers
- The explanation should accurately reflect the character's semantics in ancient times
- Can include construction explanations, but the focus is on semantic explanation
- Output concisely and clearly, no more than 150 words

## Example Output:
"Number one, resembling a horizontal line, representing the smallest positive integer or beginning."
"City, resembling a square city shape, representing city or capital."
"Person, resembling a person standing sideways, representing human or person."
"Water, resembling flowing water, representing water or water-related things."
"Rest, from person + tree, resembling a person resting against a tree, representing rest or stop."
"River, from water + sound, meaning component water indicates water-related, sound component indicates pronunciation, referring to river."

## Important Notes:
- Only use visual analysis, do not use any external knowledge
- Focus on the main character image for analysis
- Output should be in English
- Be concise and accurate"""

        else:
            # KG Pipeline - system prompt with knowledge base support
            variant_system_prompt = """You are an oracle bone script expert, specializing in analyzing oracle bone characters' complete meanings.

## Core Task:
Analyze oracle bone character images, combine radical information and database retrieval information to understand and explain the complete meaning of this character.

## Information Priority:
1. **Database Information**: If database retrieval results are provided, please explain based on this information
2. **Visual Analysis**: Combine image features to verify and supplement database information
3. **Radical Information**: Use radical information to understand the character's construction principles

## Image Order Description:
- The first image is the main character image, which is the primary object you need to analyze
- Subsequent images are radical images to help you understand the character's construction principles
- Always focus on the first image (main character) as the primary analysis object

## Important Understanding:
- Database information is usually the most accurate, please prioritize using it
- Radical information only helps you understand the character's construction principles, not to explain the radicals themselves
- You need to analyze what concept the entire character expresses based on all available information
- Output should be the complete explanation of this character, not the explanation of radicals

## Analysis Steps:
1. **Database Information Analysis**: If database retrieval results are provided, please carefully analyze this information
2. **Visual Verification**: Combine image features to verify the accuracy of database information
3. **Construction Understanding**: Combine radical images to understand how the character is constructed
4. **Semantic Inference**: Based on database information and construction principles, infer what meaning this character expressed in ancient times
5. **Comprehensive Explanation**: Give a complete and accurate explanation of the character

## Analysis Principles:
- Pictographic characters: Directly depict the shape of things, such as "人" (person) resembling human form, "日" (sun) resembling sun shape
- Ideographic characters: Express meaning through component combination, such as "休" (rest) from person + tree, indicating a person resting against a tree
- Phonetic characters: Meaning component + sound component, such as "河" (river) from water + sound
- Indicative characters: Use symbols to indicate position or state, such as "上" (up), "下" (down)

## Output Requirements:
- Directly output the complete explanation of the character, without any prefix or format markers
- If database information is sufficient, please explain based on database information
- The explanation should accurately reflect the character's semantics in ancient times
- Can include construction explanations, but the focus is on semantic explanation
- Output concisely and clearly, no more than 150 words

## Example Output:
"Number one, resembling a horizontal line, representing the smallest positive integer or beginning."
"City, resembling a square city shape, representing city or capital."
"Person, resembling a person standing sideways, representing human or person."
"Water, resembling flowing water, representing water or water-related things."
"Rest, from person + tree, resembling a person resting against a tree, representing rest or stop."
"River, from water + sound, meaning component water indicates water-related, sound component indicates pronunciation, referring to river."

## Database Information Usage Instructions:
- If database retrieval results are provided, please explain based on this information
- Combine visual features and radical information to generate accurate and complete explanations
- Keep it concise and clear, avoid verbosity"""

        # Create user message
        if custom_prompt:
            # Use custom prompt
            user_text = custom_prompt
        else:
            # Use default prompt
            user_text = f"""Please analyze the complete meaning of this oracle bone character.

Available radical information: {radical_list}

Important reminders:
- The first image is the main character image, which is the primary object you need to analyze
- Subsequent images are radical images to help you understand the character's construction principles
- Radical information only helps you understand the character's construction principles
- You need to analyze what concept the entire character expresses, not explain the radicals
- Based on visual features and construction principles, give the complete explanation of the character

Please directly output the character's explanation without any format markers."""
        
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
        
        # Add radical images - ensure correct order
        for i, radical_base64 in enumerate(base64_radical):
            if i < 2:  # Add at most 2 radical images
                user_messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{radical_base64}"}
                })

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", variant_system_prompt),
            ("user", user_messages)
        ])

        # Execute - fix: pass correct parameters
        print(f"🔄 Starting LLM call...")
        print(f"📸 Main image size: {len(base64_image)} characters")
        print(f"📸 Radical image count: {len(base64_radical)}")
        print(f"📝 Radical list: {radical_list}")
        print(f"📝 Message structure: text + main image + {len(base64_radical)} radical images")
        
        chain = (prompt | model)
        
        # Fix: pass necessary parameters
        response = chain.invoke({
            "base64_image": base64_image,
            "radical_list": radical_list
        })
        
        # Check response
        if not response:
            print("❌ LLM returned empty response object")
            return "LLM returned empty response object", False
            
        if not hasattr(response, 'content'):
            print("❌ LLM response object has no content attribute")
            print(f"Response object type: {type(response)}")
            print(f"Response object content: {response}")
            return "LLM response object format error", False
            
        if not response.content or not response.content.strip():
            print("❌ LLM returned empty content")
            return "LLM returned empty content", False
            
        print(f"✅ LLM call successful, response length: {len(response.content)}")
        print(f"📝 LLM response preview: {response.content[:100]}...")
        
        return response.content, False  # Return (output, used_baseline_fallback)
        
    except Exception as e:
        print(f"❌ Exception occurred during LLM call: {e}")
        import traceback
        traceback.print_exc()
        return f"LLM call exception: {e}", False
