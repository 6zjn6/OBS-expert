from KG_construct_ENG import KG_construct_new
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
from chatgpt_rag_ENG import (
    chat_with_gpt_variant_explanation_ENG,
    search_exact_character,
    search_character_by_radical,
    search_radical_explanation
)
import argparse
import os
import random
from pathlib import Path
import sys

# 添加父目录到路径，以便导入robust_csv_reader
sys.path.append('..')
from robust_csv_reader import robust_read_csv

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

    # Read data with robust CSV reading
    print(f"   Reading data from: {csv_file}")
    try:
        df = robust_read_csv(csv_file, expected_columns=2)  # character_explanations.csv has 2 columns
        print(f"   Total data: {len(df)} characters")
    except Exception as e:
        print(f"❌ Failed to read data file: {e}")
        raise e

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

def build_kg_with_training_data(train_df):
    """
    Build knowledge graph using training set data
    """
    print("🔨 Building knowledge graph using training set data...")

    # Create temporary seen data file
    # 基于文件位置解析到 experiments/data
    base_data_dir = Path(__file__).resolve().parents[2] / 'data'
    temp_seen_file = str(base_data_dir / 'character_explanations_temp_seen.csv')
    train_df.to_csv(temp_seen_file, index=False, encoding='utf-8-sig')

    try:
        # Temporarily modify environment variable to let KG construction use seen data
        original_default = str(base_data_dir / 'character_explanations.csv')
        original_csv = os.environ.get('CHARACTER_CSV_FILE', original_default)
        os.environ['CHARACTER_CSV_FILE'] = temp_seen_file

        # Build KG
        KG_construct_new()
        print("✅ Knowledge graph construction successful!")

        # Restore original environment variable
        os.environ['CHARACTER_CSV_FILE'] = original_csv

    except Exception as e:
        print(f"❌ Knowledge graph construction failed: {e}")
        # Restore original environment variable
        os.environ['CHARACTER_CSV_FILE'] = original_csv
        raise e
    finally:
        # Clean up temporary file
        if os.path.exists(temp_seen_file):
            os.remove(temp_seen_file)

def get_possible_radical(radical_image, class_prototypes, train_classes, model, all_radical_list, std, mean, topk=5):
    """Use PrototypeClassifier to predict radicals"""
    return get_possible_radical_prototype(radical_image, class_prototypes, train_classes, model, all_radical_list, std, mean, topk)

def get_radical_images_from_img_zi(character, image_name):
    """Get radical images from img_zi folder"""
    img_zi_dir = str((Path(__file__).resolve().parents[2] / 'data' / 'img_zi' / character))
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

def generate_explanation_from_database(character, radical_list):
    """Generate character explanation based on database search, prioritize using database information"""
    print(f"    🔍 Starting database search...")

    # ⚠️ Important: Testing phase prohibits exact character search to avoid data leakage!
    # 1. Prohibit exact character search (this would cause data leakage)
    # exact_result = search_exact_character(character)
    # if "Found character" in exact_result and "not found" not in exact_result:
    #     print(f"    ✅ Found exact match: {exact_result}")
    #     return exact_result

    print(f"    ⚠️  Testing phase prohibits exact character search to avoid data leakage")

    # 2. If no exact match, try searching through radicals
    if radical_list:
        print(f"    🔍 Searching through radicals: {radical_list}")
        radical_results = []

        for radical in radical_list:
            # Search radical explanation
            radical_explanation = search_radical_explanation(radical)
            if "not found" not in radical_explanation:
                radical_results.append(f"Radical '{radical}': {radical_explanation}")

            # Search characters containing this radical
            character_by_radical = search_character_by_radical(radical)
            if "not found" not in character_by_radical:
                radical_results.append(f"Characters containing radical '{radical}': {character_by_radical}")

        if radical_results:
            combined_result = "Based on radical analysis:\n" + "\n".join(radical_results)
            print(f"    ✅ Found relevant information through radicals")
            return combined_result

    # 3. If nothing found, return empty string for LLM to handle
    print(f"    ⚠️ No relevant information found in database, returning empty string for LLM to handle")
    return ""

def process_test_characters_kg_only(test_df, force_restart=False):
    """Process test set characters, using only KG pipeline to generate oracle bone character explanations"""

    # Get PrototypeClassifier model
    print('----------Getting PrototypeClassifier model-----------')
    model, class_prototypes, train_classes, std, mean = get_prototype_model()
    if model is None:
        print("❌ Unable to get PrototypeClassifier model, exiting")
        return

    model.eval()

    # Get all radical list with robust CSV reading
    print("📖 Reading radical explanation CSV file...")
    
    # 导入强大的CSV读取器
    import sys
    sys.path.append('..')
    from robust_csv_reader import robust_read_csv
    
    try:
        base_data_dir = Path(__file__).resolve().parents[2] / 'data'
        radical_df_all = robust_read_csv(str(base_data_dir / 'radical_explanation.csv'), expected_columns=4)
        print(f"✅ Radical CSV reading successful: {radical_df_all.shape}")
    except Exception as e:
        print(f"❌ Failed to read radical explanation CSV file: {e}")
        print("❌ Unable to continue without radical data")
        return

    all_radical_list = radical_df_all['Radical'].unique().tolist()

    # Use test set data
    All_zi = test_df
    print(f"Successfully read test set data: {len(All_zi)} characters")

    # Create output directory
    output_dir = 'English_version/exp3_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get LLM model name for filename, consistent with chatgpt_rag_ENG.py get_llm()
    # KG pipeline CSV file path
    kg_file = f'{output_dir}/test_set_kg.csv'

    # Check if force restart
    if force_restart:
        if os.path.exists(kg_file):
            os.remove(kg_file)
            print(f"🗑️ Force restart, deleted existing result file: {kg_file}")
        print("🆕 Will start processing all characters from scratch")

    # Check if result file exists, support resume
    processed_characters = set()
    start_index = 0

    if os.path.exists(kg_file):
        try:
            df = robust_read_csv(kg_file)
            if len(df) > 0:
                # Filter out header row, only count actual data
                data_rows = df[df['Character'] != 'Character']  # Exclude header
                if len(data_rows) > 0:
                    processed_chars = set(data_rows['Character'].tolist())
                    processed_characters.update(processed_chars)
                    print(f"✅ Found existing result file: {kg_file}, containing {len(processed_chars)} characters")
                else:
                    print(f"⚠️ File exists but only has header: {kg_file}")
        except Exception as e:
            print(f"⚠️ Failed to read file: {kg_file}, error: {e}")

    # Calculate number of characters to skip
    if processed_characters:
        print(f"📊 Processed characters: {len(processed_characters)}")
        print(f"📊 Remaining characters: {len(All_zi) - len(processed_characters)}")

        # Find first unprocessed character index
        all_characters = All_zi['Character'].tolist()
        for i, char in enumerate(all_characters):
            if char not in processed_characters:
                start_index = i
                break
        else:
            print("🎉 All characters have been processed!")
            return
    else:
        print("🆕 Starting fresh processing, creating new result file")
        # Create CSV file and write header
        with open(kg_file, mode='w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(["Character", "Ground_Truth", "LLM_Output", "Pipeline"])

    # Read ground truth data (using English explanation data)
    try:
        gt_df = robust_read_csv(str(base_data_dir / 'character_explanations.csv'), expected_columns=2)  # character_explanations.csv has 2 columns
    except Exception as e:
        print(f"❌ Failed to read ground truth data: {e}")
        return

    # Process each character
    all_characters = All_zi['Character'].tolist()

    # Update processed_characters set to ensure correct checking in loop
    if processed_characters:
        print(f"🔄 Resume mode: starting from character {start_index + 1}")
    else:
        print(f"🆕 Fresh start mode: starting from character 1")

    print(f"🚀 Starting from character {start_index + 1}, total {len(all_characters)} characters")

    for i, zi in enumerate(all_characters):
        # Resume: skip already processed characters
        if zi in processed_characters:
            print(f"  ⏭️ Skipping already processed character: {zi}")
            continue
        print(f"\n------------Processing character {i+1}/{len(all_characters)}: {zi}")

        img_zi_dir = str((Path(__file__).resolve().parents[2] / 'data' / 'img_zi' / zi))
        if not os.path.exists(img_zi_dir):
            print(f"❌ Character {zi} image directory does not exist: {img_zi_dir}")
            continue

        cnt = 0  # Process at most 5 images per character
        for file in os.listdir(img_zi_dir):
            if cnt >= 5:
                break

            if not file.endswith('.jpg'):
                continue

            image_name = file[:-4]  # Remove .jpg suffix
            image_path = os.path.join(img_zi_dir, file)

            print(f"  Processing image: {file}")

            # Get radical images
            radical_images, radical_image_paths, target_radical = get_radical_images_from_img_zi(zi, image_name)

            if not radical_images:
                print(f"  Skip {file} - no radical images found")
                continue

            print(f"  Found {len(radical_images)} radicals: {target_radical}")

            # Use PrototypeClassifier to predict possible radicals
            possible_radicals = []
            best_radicals = []  # Only keep the most likely radical for each radical image

            for radical_image in radical_images:
                possible_radical = get_possible_radical(radical_image, class_prototypes, train_classes,
                                                      model, all_radical_list, std, mean)
                possible_radicals.append(possible_radical)

                # Only take the most similar radical
                if possible_radical and len(possible_radical) > 0:
                    best_radical = possible_radical[0]  # Take the first one (highest similarity)
                    best_radicals.append(best_radical)

            print(f"  Predicted radicals: {possible_radicals}")
            print(f"  Most likely radicals: {best_radicals}")

            # Get ground truth
            gt_row = gt_df[gt_df['Character'] == zi]
            if len(gt_row) == 0:
                print(f"  Skip {zi} - not found in ground truth")
                continue

            ground_truth = gt_row.iloc[0]['Explanation']
            print(f"  Ground Truth: {ground_truth}")

            # KG Pipeline - LLM uses PrototypeClassifier to get radicals and calls KG
            print(f"  🔄 KG Pipeline: LLM uses PrototypeClassifier to get radicals and calls KG...")
            kg_output_cleaned = None

            # Prioritize using database search, then provide information to LLM for optimization
            print(f"    🔍 Prioritizing database search...")
            database_output = generate_explanation_from_database(zi, best_radicals)

            # Pass database information as context to LLM
            enhanced_prompt = f"""
Database information: {database_output}

Based on image analysis, radical information {best_radicals} and database information, directly output the explanation of character "{zi}".

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

Please directly output the explanation of character "{zi}":"""

            # Retry mechanism
            for attempt in range(3):
                try:
                    print(f"    🔄 KG attempt {attempt + 1}...")
                    kg_output, _ = chat_with_gpt_variant_explanation_ENG(
                        image_path, radical_image_paths, best_radicals, enhanced_prompt, is_baseline=False
                    )
                    kg_output_cleaned = clean_llm_output(kg_output, zi, is_baseline=False)

                    # Check if output is valid
                    if kg_output_cleaned and len(kg_output_cleaned.strip()) > 10:
                        print(f"    ✅ KG output successful: {kg_output_cleaned}")
                        break
                    else:
                        print(f"    ⚠️ KG output too short, retrying...")
                        if attempt < 2:
                            import time
                            time.sleep(1)

                except Exception as e:
                    print(f"    ❌ KG attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        import time
                        time.sleep(1)

            # If all retries fail, use empty output
            if not kg_output_cleaned or len(kg_output_cleaned.strip()) <= 10:
                kg_output_cleaned = ""
                print(f"    ⚠️ KG all retries failed, using empty output")

            # Save KG result
            print(f"  💾 Saving KG result...")

            # KG result
            with open(kg_file, mode='a', newline='', encoding='utf-8-sig', errors='ignore') as file:
                writer = csv.writer(file)
                writer.writerow([zi, ground_truth, kg_output_cleaned, "KG"])

            # Update processed characters set for resume
            processed_characters.add(zi)

            # Show progress
            remaining = len(all_characters) - len(processed_characters)
            print(f"  ✅ Character {zi} processing completed, {remaining} characters remaining")

            cnt += 1

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Oracle Bone Script Explanation System - KG Pipeline (English Version)')
    parser.add_argument('--shuffle', action='store_true', default=True,
                       help='Whether to shuffle oracle bone character order (default: True)')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                       help='Do not shuffle oracle bone character order')
    parser.add_argument('--llm-model', type=str, default=None, help='LLM model name')
    parser.add_argument('--llm-base-url', type=str, default=None, help='LLM API base URL')
    parser.add_argument('--llm-api-key', type=str, default=None, help='LLM API key')
    parser.add_argument('--llm-temperature', type=float, default=None, help='LLM temperature parameter')
    parser.add_argument('--llm-max-tokens', type=int, default=None, help='LLM max tokens')
    parser.add_argument('--llm-enable-thinking', action='store_true', help='Enable thinking mode (only supports reasoning models)')
    parser.add_argument('--llm-auto-downgrade', action='store_true', default=True, help='Auto downgrade when model does not support thinking (default enabled)')
    parser.add_argument('--llm-no-auto-downgrade', action='store_true', help='Disable auto downgrade, force thinking mode')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio for building KG (default: 0.7)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed to ensure data splitting reproducibility (default: 42)')
    parser.add_argument('--force-rebuild-kg', action='store_true', default=True, help='Force rebuild knowledge graph, delete existing content (default: True)')
    parser.add_argument('--no-force-rebuild-kg', dest='force_rebuild_kg', action='store_false', help='Do not force rebuild knowledge graph, skip if exists')
    parser.add_argument('--test-file', type=str, default=None, help='Specify test set file path, if not specified will auto-find')
    parser.add_argument('--force-restart', action='store_true', help='Force restart, delete existing result files')
    parser.add_argument('--resume', action='store_true', default=True, help='Resume mode (default: True)')

    args = parser.parse_args()

    # Runtime override environment variables for chatgpt_rag_ENG.get_llm() to use
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

    # Ensure GPU result consistency
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data splitting: 7/10 for building KG, 3/10 as test set
    print('📊 Starting data splitting...')
    csv_file = str((Path(__file__).resolve().parents[2] / 'data' / 'character_explanations.csv'))

    # Initialize train_df and test_df
    train_df = None
    test_df = None
    
    # Check if test file is specified
    if args.test_file:
        test_file = args.test_file
        print(f'✅ Using specified test set file: {test_file}')
        test_df = robust_read_csv(test_file)
        print(f'   Test set count: {len(test_df)} characters')
        
        # If using specified test file, we need to create train_df from the remaining data
        print('⚠️ Using specified test file - need to create training data for KG construction')
        all_data = robust_read_csv(csv_file, expected_columns=2)
        test_characters = set(test_df['Character'].tolist())
        train_data = all_data[~all_data['Character'].isin(test_characters)]
        train_df = train_data.copy()
        print(f'   Training set count: {len(train_df)} characters')
    else:
        # Check if split test set file exists
        test_file = csv_file.replace('.csv', '_unseen.csv')
        if os.path.exists(test_file):
            print(f'✅ Found split test set file: {test_file}')
            test_df = robust_read_csv(test_file)
            print(f'   Test set count: {len(test_df)} characters')
            
            # Load corresponding training set
            train_file = csv_file.replace('.csv', '_seen.csv')
            if os.path.exists(train_file):
                print(f'✅ Found corresponding training set file: {train_file}')
                train_df = robust_read_csv(train_file, expected_columns=2)
                print(f'   Training set count: {len(train_df)} characters')
            else:
                print('⚠️ Training set file not found, creating from remaining data')
                all_data = robust_read_csv(csv_file, expected_columns=2)
                test_characters = set(test_df['Character'].tolist())
                train_data = all_data[~all_data['Character'].isin(test_characters)]
                train_df = train_data.copy()
                print(f'   Training set count: {len(train_df)} characters')
        else:
            print(f'⚠️ No split test set file found, re-splitting data')
            # Use same data splitting logic as run3_ENG.py
            train_df, test_df = split_data_for_kg_and_test(csv_file, train_ratio=args.train_ratio, random_seed=args.random_seed)
            print(f'   Training set count: {len(train_df)} characters')
            print(f'   Test set count: {len(test_df)} characters')

    # Decide whether to force rebuild knowledge graph based on command line arguments
    if args.force_rebuild_kg:
        print('🗑️ Force deleting existing knowledge graph...')
        try:
            from py2neo import Graph
            graph = Graph("bolt://localhost:7687", auth=("neo4j", "87654321"))

            # Check existing node count
            node_count = len(graph.nodes)
            if node_count > 0:
                print(f'⚠️ Found existing knowledge graph with {node_count} nodes')
                print('🗑️ Clearing knowledge graph...')

                # Delete all nodes and relationships
                graph.run("MATCH (n) DETACH DELETE n")
                print('✅ Knowledge graph completely cleared')
            else:
                print('ℹ️ Knowledge graph is empty, no need to clear')

        except Exception as e:
            print(f'⚠️ Failed to clear knowledge graph: {e}')
            print('🔨 Continuing to build new knowledge graph...')

        # Build new knowledge graph
        print('🔨 Starting to build new knowledge graph...')
        build_kg_with_training_data(train_df)
    else:
        # Check if knowledge graph already exists, skip building if it does
        print('🔍 Checking existing knowledge graph...')
        try:
            from py2neo import Graph
            graph = Graph("bolt://localhost:7687", auth=("neo4j", "87654321"))
            node_count = len(graph.nodes)
            if node_count > 0:
                print(f'✅ Knowledge graph already exists with {node_count} nodes, skipping build')
                print('⚠️ Note: Using existing KG may cause data leakage!')
            else:
                print('🔨 KG build starting!')
                # Build KG using training set data
                build_kg_with_training_data(train_df)
        except Exception as e:
            print(f'⚠️ Failed to check knowledge graph: {e}')
            print('🔨 KG build starting!')
            # Build KG using training set data
            build_kg_with_training_data(train_df)

    # Cache functionality removed
    print('ℹ️ Cache functionality has been removed')

    # Check img_zi directory
    if not os.path.exists(str((Path(__file__).resolve().parents[2] / 'data' / 'img_zi'))):
        print("../data/img_zi directory does not exist")
        exit(1)

    # Process test set characters
    print('🚀 Starting to process test set characters...')
    process_test_characters_kg_only(test_df, force_restart=args.force_restart)

    print(f'\n🎉 All character processing completed!')

    # Output result statistics
    output_dir = 'English_version/exp3_output'
    # KG pipeline result file
    kg_file = f'{output_dir}/test_set_kg.csv'

    print("🎉 KG pipeline result saved to English_version/exp3_output directory")
    print(f"📊 Test set character count: {len(test_df)}")

    # Count KG results
    if os.path.exists(kg_file):
        try:
            df = robust_read_csv(kg_file)
            print(f"📊 KG Pipeline total samples: {len(df)}")
        except Exception as e:
            print(f"❌ Failed to read KG result file: {e}")
    else:
        print(f"❌ KG Pipeline result file does not exist: {kg_file}")

    print(f"\n📁 Output file:")
    print(f"  KG: {kg_file}")
