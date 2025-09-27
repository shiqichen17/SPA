"""
Script for generating trajectories with worldmodel mode for SFT training.
"""

import os
import pdb
import json
import numpy as np
import time
import logging
import sys
import re
from datetime import datetime
import hydra
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from SPA_agent.agent_proxy import LLMAgentProxy, VllmWrapperWg
from verl import DataProto
import pdb

def init_logging(to_file_only=False, log_dir="log"):
    """Set up logging: redirect stdout/stderr to file and optionally keep console output."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"debug_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handlers = [logging.FileHandler(log_file, mode='w', encoding='utf-8')]
    if not to_file_only:
        handlers.append(logging.StreamHandler(sys.__stdout__))

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

    # Redirect print and errors to logging
    class StreamToLogger:
        def __init__(self, level): self.level = level
        def write(self, message):
            message = message.strip()
            if message: self.level(message)
        def flush(self): pass

    sys.stdout = StreamToLogger(logging.info)
    sys.stderr = StreamToLogger(logging.error)


def convert_to_sft_format_add_worldmodel(trajectories, tokenizer):
    """Convert trajectories to SFT format by replacing predicted states with real states.
    
    Each turn has the structure:
    <|imstart|> xxxxx <next_state> real_state </next_state> <|imend|>
    """
    sft_data = []
    index = 0
    for traj in trajectories:
        # Get the messages and states
        messages = traj['messages_list']
        real_states = traj['real_states']
        
        if real_states is None:
            continue
            
        # Process conversation
        input_ids = []
        labels = []
        
        # Add system message
        system_message = messages[0]
        system_tokens = tokenizer.encode('<|im_start|>system\n' + system_message['content'] + '\n<|im_end|>', add_special_tokens=False)
        input_ids.extend(system_tokens)
        labels.extend([-100] * len(system_tokens))
        
        # Process each turn (user + assistant)
        for turn_idx in range(len(messages)//2):
            user_idx = 1 + turn_idx * 2
            assistant_idx = user_idx + 1
            cur_state_idx = turn_idx
            next_state_idx = turn_idx + 1
            
            # Add user message
            if user_idx < len(messages):
                user_tokens = tokenizer.encode('<|im_start|>user\n' + messages[user_idx]['content'] + '\n<|im_end|>', add_special_tokens=False)
                input_ids.extend(user_tokens)
                labels.extend([-100] * len(user_tokens))
            
            # Add assistant message with real state
            if assistant_idx < len(messages):
                assistant_message = messages[assistant_idx]['content']

                # Get real states for replacement
                cur_state = ""
                if cur_state_idx < len(real_states) and real_states[cur_state_idx] is not None:
                    cur_state = real_states[cur_state_idx]

                next_state = ""
                if next_state_idx < len(real_states) and real_states[next_state_idx] is not None:
                    next_state = real_states[next_state_idx]

                # Replace predicted state with real state using regex
                def replace_states_in_message(msg, obs, pred):
                    
                    pattern = r'(<observation>)(.*?)(</observation>)'
                    msg = re.sub(pattern, f'\\1{obs}\\3', msg, flags=re.DOTALL)
                    pattern = r'(<prediction>)(.*?)(</prediction>)'
                    msg = re.sub(pattern, f'\\1{pred}\\3', msg, flags=re.DOTALL)
                    # import pdb; pdb.set_trace()
                    return msg

                assistant_message = replace_states_in_message(assistant_message, cur_state, next_state)
                
                # Update the assistant message in the original messages list
                messages[assistant_idx]['content'] = assistant_message
                
                # Add the complete message
                assistant_tokens = tokenizer.encode('<|im_start|>assistant\n' + assistant_message + '\n<|im_end|>', add_special_tokens=False)
                input_ids.extend(assistant_tokens)
                
                # Create labels with -100 for non-state positions
                turn_labels = [-100] * len(assistant_tokens)
                
                # Find state positions
                state_start_tokens = tokenizer.encode('<observation>', add_special_tokens=False)
                state_end_tokens = tokenizer.encode('</observation>', add_special_tokens=False)
                
                # Find indices of state tags
                start_indices = [i for i in range(len(assistant_tokens)) if assistant_tokens[i:i+len(state_start_tokens)] == state_start_tokens]
                end_indices = [i for i in range(len(assistant_tokens)) if assistant_tokens[i:i+len(state_end_tokens)] == state_end_tokens]
                
                if start_indices and end_indices:
                    start_idx = start_indices[0]
                    end_idx = end_indices[0] + len(state_end_tokens)
                    # Only keep values between state tags
                    turn_labels[start_idx:end_idx] = assistant_tokens[start_idx:end_idx]
                
                labels.extend(turn_labels)
        index += 1
        # Add to SFT data
        
        
        sft_data.append({
            'id':index,
            'messages_list': messages,  # Use the modified messages list
            
        })
    
    return sft_data

@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config):
    """Generate trajectories with worldmodel mode and save them for SFT training."""
    # Initialize logging
    init_logging(to_file_only=False)
    import argparse
    
    # Set environment variables
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", str(config.system.CUDA_VISIBLE_DEVICES))
    
    
    model_path = config.model_path
    config.actor_rollout_ref.model.path = model_path
    print(f"Loading tokenizer from {config.actor_rollout_ref.model.path}")
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    actor_wg = VllmWrapperWg(config, tokenizer)
    proxy = LLMAgentProxy(config, actor_wg, tokenizer)
    
    # Create output directory
    output_dir = os.getenv("OUTPUT_DIR")
    # Ensure OUTPUT_DIR environment variable is provided; otherwise raise an error
    if output_dir is None or str(output_dir).strip() == "":
        raise RuntimeError("Environment variable 'OUTPUT_DIR' is not set. "
                           "Please export OUTPUT_DIR before running this script.")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate multiple batches of trajectories
    num_batches = os.getenv("BT_NUM")  # You can adjust this number
    all_trajectories = []
    
    total_start_time = time.time()
    
    for batch_idx in range(int(num_batches)):
        batch_start_time = time.time()
        print(f"Generating batch {batch_idx + 1}/{num_batches}")
        
        # Generate trajectories
        meta_info = {
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            'validate': True,
        }
        test_gen_batch = DataProto(batch=None, non_tensor_batch=None, meta_info=meta_info)
        
        # Get trajectories with both predicted and real states
        test_batch, rollout_states, all_states = proxy.rollout(test_gen_batch, val=False)
        
        # Process and store trajectories
        for i in range(len(test_batch.batch['responses'])):
            success_flag = (
                rollout_states[i].get('metrics', {}).get('SimpleSokoban/success', 0.0) == 1.0
                or rollout_states[i].get('metrics', {}).get('FrozenLake/success', 0.0) == 1.0
            )
            # If neither environment reports success, skip this trajectory
            # if not success_flag:
            #     continue
            trajectory = {
            'messages_list': test_batch.non_tensor_batch['messages_list'][i],
            'real_states': all_states[i],
            'rewards': test_batch.batch['rm_scores'][i].cpu().tolist(),
            }
            all_trajectories.append(trajectory)
            

        
        batch_time = time.time() - batch_start_time
        print(f"Batch {batch_idx + 1} completed in {batch_time:.2f} seconds")
        print(f"Generated {len(test_batch.batch['responses'])} trajectories in this batch")
    
    total_time = time.time() - total_start_time
    print(f"\nTotal generation time: {total_time:.2f} seconds")
    
    
    # Convert to SFT format
    print("\nConverting trajectories to SFT format...")
    if config.ctx_manager.mode == 'add_worldmodel':
        sft_data = convert_to_sft_format_add_worldmodel(all_trajectories, tokenizer)
    
    # Save trajectories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw trajectories
    raw_output_file = os.path.join(output_dir, f"raw_trajectories_{timestamp}.json")
    
    with open(raw_output_file, 'w') as f:
        json.dump(sft_data, f, indent=2)
    print(f"Saved raw trajectories to {raw_output_file}")

    # 直接转 DataFrame 并划分 train/val
    import pandas as pd
    from sklearn.model_selection import train_test_split
    rows = []
    DEFAULT_DATA_SOURCE = "sokoban"
    DEFAULT_ABILITY = "bfs"
    DEFAULT_REWARD_MODEL = "{'ground_truth': {'numbers': array([0, 0]), 'target': 0}, 'style': 'rule'}"
    DEFAULT_EXTRA_INFO = "{'index': 100016, 'split': 'train'}"


    for sample in sft_data:
        messages = sample["messages_list"]
        for i, msg in enumerate(messages):
            if msg["role"] == "assistant":
                prompt_list = np.array(messages[:i])
                print(prompt_list.dtype, prompt_list.shape)
                rows.append({
                    'data_source': DEFAULT_DATA_SOURCE,
                    # 'prompt': json.dumps(prompt_list, ensure_ascii=False),
                    'prompt': prompt_list,
                    'response': msg['content'],
                    'ability': DEFAULT_ABILITY,
                    'reward_model': DEFAULT_REWARD_MODEL,
                    'extra_info': DEFAULT_EXTRA_INFO,
                })
    df = pd.DataFrame(rows)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    train_csv = os.path.join(output_dir, 'wm_train.csv')
    val_csv = os.path.join(output_dir, 'wm_val.csv')
    train_parquet = os.path.join(output_dir, 'wm_train.parquet')
    val_parquet = os.path.join(output_dir, 'wm_val.parquet')
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    train_df.to_parquet(train_parquet, index=False)
    val_df.to_parquet(val_parquet, index=False)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"Saved to {train_csv}, {val_csv}, {train_parquet}, {val_parquet}")
    print(f"\nGenerated {len(all_trajectories)} total trajectories")
    print(f"Converted to {len(sft_data)} SFT training examples")

if __name__ == "__main__":
    main()