#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities for English version oracle bone script processing
"""

import pandas as pd
import os
import csv
import re
import random
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

# Global variables
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def split_data_for_kg_and_test(csv_file, train_ratio=0.7, random_seed=42):
    """
    Split data into training set (for building KG) and test set (for testing LLM)
    
    Args:
        csv_file: CSV file path
        train_ratio: Training set ratio, default 0.7 (7/10)
        random_seed: Random seed to ensure reproducibility
    
    Returns:
        train_df: Training set dataframe
        test_df: Test set dataframe
    """
    print(f"📊 Starting data splitting...")
    print(f"   Training set ratio: {train_ratio:.1%}")
    print(f"   Test set ratio: {1-train_ratio:.1%}")
    print(f"   Random seed: {random_seed}")
    
    # Read data
    df = pd.read_csv(csv_file)
    print(f"   Total data: {len(df)} characters")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Randomly shuffle data
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Calculate split point
    split_point = int(len(df_shuffled) * train_ratio)
    
    # Split data
    train_df = df_shuffled.iloc[:split_point].copy()
    test_df = df_shuffled.iloc[split_point:].copy()
    
    print(f"   Training set count: {len(train_df)} characters")
    print(f"   Test set count: {len(test_df)} characters")
    
    # Save split data
    train_file = csv_file.replace('.csv', '_seen.csv')
    test_file = csv_file.replace('.csv', '_unseen.csv')
    
    train_df.to_csv(train_file, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_file, index=False, encoding='utf-8-sig')
    
    print(f"   Training set saved to: {train_file}")
    print(f"   Test set saved to: {test_file}")
    
    return train_df, test_df

def get_radical_images_from_img_zi(character, image_name):
    """Get radical images from img_zi folder"""
    img_zi_dir = f'../data/img_zi/{character}'
    radical_images = []
    radical_image_paths = []
    target_radical = []
    
    if not os.path.exists(img_zi_dir):
        print(f"Directory does not exist: {img_zi_dir}")
        return [], [], []
    
    # Find corresponding image folder
    for file in os.listdir(img_zi_dir):
        if file.endswith('.jpg') and image_name in file:
            base_name = file[:-4]  # Remove .jpg suffix
            print(f"Found image: {file}")
            
            # Find corresponding radical images
            for radical_file in os.listdir(img_zi_dir):
                if radical_file.endswith('.png') and base_name in radical_file and '_' in radical_file:
                    # Extract radical name
                    radical_name = radical_file.split('_')[-1][:-4]  # Remove .png suffix
                    target_radical.append(radical_name)
                    
                    # Read radical image
                    image_path = os.path.join(img_zi_dir, radical_file)
                    image = Image.open(image_path)
                    radical_image_paths.append(image_path)
                    
                    # Convert to RGB format
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
    """Clean LLM output, keep only explanation content, if empty return empty string for LLM to handle"""
    print(f"    🔍 Starting output cleaning, original output: '{output_text}'")
    
    if not output_text or output_text.strip() == "":
        print(f"    ⚠️ LLM output is empty, returning empty string")
        return ""
    
    # Directly clean output, as LLM now directly outputs explanation
    cleaned_output = output_text.strip()
    print(f"    🔍 After removing leading/trailing whitespace: '{cleaned_output}'")
    
    # Remove possible format markers (but keep content and punctuation)
    # Only remove leading pure format markers, keep content
    original_cleaned = cleaned_output
    cleaned_output = re.sub(r'^[-\*•\s]+', '', cleaned_output)
    if cleaned_output != original_cleaned:
        print(f"    🔍 Removed leading format markers: '{original_cleaned}' -> '{cleaned_output}'")
    
    # Only remove trailing pure format markers, keep content
    original_cleaned = cleaned_output
    cleaned_output = re.sub(r'[-\*•\s]+$', '', cleaned_output)
    if cleaned_output != original_cleaned:
        print(f"    🔍 Removed trailing format markers: '{original_cleaned}' -> '{cleaned_output}'")
    
    # Check if cleaned output is empty
    if not cleaned_output:
        print(f"    ⚠️ Cleaned output is empty, returning empty string")
        return ""
    
    # If output is too long, only keep first 300 characters (explanations usually need more text)
    if len(cleaned_output) > 300:
        cleaned_output = cleaned_output[:300] + "..."
    
    print(f"    ✅ Output cleaning successful, length: {len(cleaned_output)}, content: '{cleaned_output}'")
    return cleaned_output

def setup_output_directory(output_dir='English_version/exp3_output'):
    """Setup output directory and return LLM model name for filename"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get LLM model name for filename, consistent with chatgpt_rag_ENG.py get_llm()
    return "output"

def check_resume_status(file_path, processed_characters=None):
    """Check if we can resume from existing results"""
    if processed_characters is None:
        processed_characters = set()
    
    start_index = 0
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                # Filter out header row, only count actual data
                data_rows = df[df['Character'] != 'Character']  # Exclude header
                if len(data_rows) > 0:
                    processed_chars = set(data_rows['Character'].tolist())
                    processed_characters.update(processed_chars)
                    print(f"✅ Found existing result file: {file_path}, containing {len(processed_chars)} characters")
                else:
                    print(f"⚠️ File exists but only has header: {file_path}")
        except Exception as e:
            print(f"⚠️ Failed to read file: {file_path}, error: {e}")
    
    return processed_characters, start_index

def create_csv_header(file_path):
    """Create CSV file with header"""
    with open(file_path, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(["Character", "Ground_Truth", "LLM_Output", "Pipeline"])

def save_result_to_csv(file_path, character, ground_truth, llm_output, pipeline):
    """Save result to CSV file"""
    with open(file_path, mode='a', newline='', encoding='utf-8-sig', errors='ignore') as file:
        writer = csv.writer(file)
        writer.writerow([character, ground_truth, llm_output, pipeline])

def print_progress(current, total, character, remaining=None):
    """Print processing progress"""
    if remaining is not None:
        print(f"  ✅ Character {character} processing completed, {remaining} characters remaining")
    else:
        print(f"\n------------Processing character {current}/{total}: {character}")

def get_baseline_prompt():
    """Get baseline pipeline prompt"""
    return """
Analyze the visual features of this oracle bone character image and directly output its explanation.

Requirements:
1. Directly output the character's explanation without any analysis process or format markers
2. Give specific meaning based on visual analysis
3. If it looks like some object, directly state what it is
4. If it represents some action, directly state what action
5. If it represents some concept, directly state what concept

Output format: Directly output explanation without other content.

Please directly output the explanation of this character:"""

def get_kg_prompt(character, best_radicals, database_output):
    """Get KG pipeline prompt"""
    return f"""
Database information: {database_output}

Based on image analysis, radical information {best_radicals} and database information, directly output the explanation of character "{character}".

Requirements:
1. Directly output the character's explanation without any analysis process or format markers
2. Give specific meaning based on visual analysis
3. If it looks like some object, directly state what it is
4. If it represents some action, directly state what action
5. If it represents some concept, directly state what concept

Output format: Directly output explanation without other content.

Examples:
For "聿": "Writing tool or writing action. From hand + bamboo, resembling holding a bamboo brush, the original form of '筆'."
For "雷": "Thunder and lightning. From rain + thunder, resembling the sound of thunder."

Please directly output the explanation of character "{character}":"""
