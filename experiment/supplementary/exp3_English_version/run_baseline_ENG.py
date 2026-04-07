import pandas as pd
import os
import torch
import argparse
from pathlib import Path
from chatgpt_rag_ENG import (
    chat_with_gpt_variant_explanation_ENG
)
from common_utils_ENG import (
    split_data_for_kg_and_test,
    get_radical_images_from_img_zi,
    clean_llm_output,
    setup_output_directory,
    check_resume_status,
    create_csv_header,
    save_result_to_csv,
    print_progress,
    get_baseline_prompt
)


def process_test_characters_baseline_only(test_df, force_restart=False):
    """Process test set characters, using only baseline pipeline to generate oracle bone character explanations"""
    
    # Use test set data
    All_zi = test_df
    print(f"Successfully read test set data: {len(All_zi)} characters")
    
    # Setup output directory and get model name
    output_dir = 'English_version/exp3_output'
    setup_output_directory(output_dir)

    # Baseline pipeline CSV file path
    baseline_file = f'{output_dir}/test_set_baseline.csv'
    
    # Check if force restart
    if force_restart:
        if os.path.exists(baseline_file):
            os.remove(baseline_file)
            print(f"🗑️ Force restart, deleted existing result file: {baseline_file}")
        print("🆕 Will start processing all characters from scratch")
    
    # Check resume status
    processed_characters, start_index = check_resume_status(baseline_file)
    
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
        create_csv_header(baseline_file)
    
    # Read ground truth data (using English explanation data)
    try:
        # 导入强大的CSV读取器
        import sys
        sys.path.append('..')
        from robust_csv_reader import robust_read_csv
        base_data_dir = Path(__file__).resolve().parents[2] / 'data'
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
            
            # Get ground truth
            gt_row = gt_df[gt_df['Character'] == zi]
            if len(gt_row) == 0:
                print(f"  Skip {zi} - not found in ground truth")
                continue
                
            ground_truth = gt_row.iloc[0]['Explanation']
            print(f"  Ground Truth: {ground_truth}")
            
            # Baseline Pipeline - LLM gets image information (including radical images) but no radical prediction
            print(f"  🔄 Baseline Pipeline: LLM gets image information (including radical images) but no radical prediction...")
            baseline_output_cleaned = None
            
            # Get baseline prompt
            baseline_prompt = get_baseline_prompt()
            
            # Retry mechanism for Baseline
            for attempt in range(3):
                try:
                    print(f"    🔄 Baseline attempt {attempt + 1}...")
                    baseline_output, _ = chat_with_gpt_variant_explanation_ENG(
                        image_path, radical_image_paths, [], baseline_prompt, is_baseline=True  # Pass radical images but not radical prediction list, mark as Baseline
                    )
                    baseline_output_cleaned = clean_llm_output(baseline_output, zi, is_baseline=True)
                    
                    # Check if output is valid
                    if baseline_output_cleaned and len(baseline_output_cleaned.strip()) > 2:
                        print(f"    ✅ Baseline output successful: {baseline_output_cleaned}")
                        break
                    else:
                        print(f"    ⚠️ Baseline output too short, retrying...")
                        if attempt < 2:
                            import time
                            time.sleep(1)
                        
                except Exception as e:
                    print(f"    ❌ Baseline attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        import time
                        time.sleep(1)
            
            # If all retries fail, keep empty output
            if not baseline_output_cleaned or len(baseline_output_cleaned.strip()) <= 2:
                print(f"    ⚠️ Baseline all retries failed, keeping empty output")
            
            # Save baseline result
            print(f"  💾 Saving baseline result...")
            save_result_to_csv(baseline_file, zi, ground_truth, baseline_output_cleaned, "Baseline")
            
            # Update processed characters set for resume
            processed_characters.add(zi)
            
            # Show progress
            remaining = len(all_characters) - len(processed_characters)
            print_progress(i+1, len(all_characters), zi, remaining)
            
            cnt += 1

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Oracle Bone Script Explanation System - Baseline Pipeline (English Version)')
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
    
    # Check if test file is specified
    if args.test_file:
        test_file = args.test_file
        print(f'✅ Using specified test set file: {test_file}')
        test_df = pd.read_csv(test_file)
        print(f'   Test set count: {len(test_df)} characters')
    else:
        # Check if split test set file exists
        test_file = csv_file.replace('.csv', '_unseen.csv')
        if os.path.exists(test_file):
            print(f'✅ Found split test set file: {test_file}')
            test_df = pd.read_csv(test_file)
            print(f'   Test set count: {len(test_df)} characters')
        else:
            print(f'⚠️ No split test set file found, re-splitting data')
            # Use same data splitting logic as run3_ENG.py
            train_df, test_df = split_data_for_kg_and_test(csv_file, train_ratio=args.train_ratio, random_seed=args.random_seed)
            print(f'   Training set count: {len(train_df)} characters')
            print(f'   Test set count: {len(test_df)} characters')
    
    print('⏭️ Baseline pipeline does not need knowledge graph construction and cache warming')
    
    # Check img_zi directory
    if not os.path.exists(str((Path(__file__).resolve().parents[2] / 'data' / 'img_zi'))):
        print("../data/img_zi directory does not exist")
        exit(1)
    
    # Process test set characters - only run baseline pipeline
    print('🚀 Starting to process test set characters - Baseline Pipeline...')
    process_test_characters_baseline_only(test_df, force_restart=args.force_restart)
    
    print(f'\n🎉 All character processing completed!')
    
    # Output result statistics
    output_dir = 'English_version/exp3_output'
    # Baseline result file
    baseline_file = f'{output_dir}/test_set_baseline.csv'
    
    print("🎉 Baseline pipeline result saved to English_version/exp3_output directory")
    print(f"📊 Test set character count: {len(test_df)}")
    
    # Count baseline results
    if os.path.exists(baseline_file):
        df = pd.read_csv(baseline_file)
        print(f"📊 Baseline Pipeline total samples: {len(df)}")
    else:
        print(f"❌ Baseline Pipeline result file does not exist: {baseline_file}")
    
    print(f"\n📁 Output file:")
    print(f"  Baseline: {baseline_file}")
