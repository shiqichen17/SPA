"""
This is the environment state manager for the LLM agent.
author: Pingyue Zhang
date: 2025-03-30
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import PIL.Image
import hydra
import random
import numpy as np
import pdb

import sys
import os
# Dynamically find RAGEN root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
ragen_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(ragen_root)
from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from ragen.utils import register_resolvers
register_resolvers()

@dataclass
class EnvStatus:
    """Status of an environment"""
    truncated: bool = False # done but not success
    terminated: bool = False # done and success
    num_actions: int = 0 # current action step (single action)
    rewards: List[float] = field(default_factory=list) # rewards for each turn
    seed: Optional[int] = None # what seed is used to reset this environment



class EnvStateManager:
    """Manager for the environment state
    The class is responsible for managing multiple (kinds of) environments
    
    """
    def __init__(self, config, mode: str = "train"):
        self.sys_config = config
        self.mode = mode
        self.config = getattr(self.sys_config.es_manager, mode)
        self.env_groups = int(self.config.env_groups)
        self.group_size = self.config.group_size
        seed_cfg = getattr(self.sys_config, "seed", None)
        if seed_cfg is not None:
            self.base_seed = seed_cfg.get(mode, None)
        else:
            self.base_seed = None
        self.seed_counter = 0
        self._init_envs()
        self.rollout_cache = None

    def _init_envs(self):
        """Initialize the environments. train_envs and val_envs are lists of envs:
        Input: tags: ["SimpleSokoban", "HarderSokoban"]; n_groups: [1, 1]; group_size: 16
        Output: envs: List[Dict], each **entry** is a dict with keys: tag, group_id, env_id, env, env_config, status
        Example: [{"tag": "SimpleSokoban", "group_id": 0, "env_id": 0, "env": env, "config": env_config, "status": EnvStatus()},
            ...
            {"tag": "SimpleSokoban", "group_id": 0, "env_id": 15 (group_size - 1), ...},
            {"tag": "HarderSokoban", "group_id": 1, "env_id": 16, ...}
            ...]
        """
        assert sum(self.config.env_configs.n_groups) == self.env_groups, f"Sum of n_groups must equal env_groups. Got sum({self.config.env_configs.n_groups}) != {self.env_groups}"
        assert len(self.config.env_configs.tags) == len(self.config.env_configs.n_groups), f"Number of tags must equal number of n_groups. Got {len(self.config.env_configs.tags)} != {len(self.config.env_configs.n_groups)}"
        self.envs = self._init_env_instances(self.config)

    def _init_env_instances(self, config):
        env_list = []
        done_groups = 0
        for tag, n_group in zip(config.env_configs.tags, config.env_configs.n_groups):
            for env_id in range(done_groups * self.group_size, (done_groups + n_group) * self.group_size):
                cfg_template = self.sys_config.custom_envs[tag]
                env_class = cfg_template.env_type
                max_actions_per_traj = cfg_template.max_actions_per_traj
                if cfg_template.env_config is None:
                    env_config = REGISTERED_ENV_CONFIGS[env_class]()
                else:
                    env_config = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)
                env_obj = REGISTERED_ENVS[env_class](env_config)
                entry = {'tag': tag, 'group_id': env_id // self.group_size, 'env_id': env_id, 
                        'env': env_obj, 'config': env_config, 'status': EnvStatus(), 'max_actions_per_traj': max_actions_per_traj}
                env_list.append(entry)
            done_groups += n_group
        return env_list

    def reset(self, seed: Optional[int] = None):
        """
        Reset the environments and get initial observation
        build up rollout cache like [{"env_id": int, "history": List[Dict], "group_id": int}, ...]
        """
        def _expand_seed(seed: int):
            seeds = [[seed + i] * self.group_size for i in range(self.env_groups)] # [[seed, ..., seed], [seed+1, ..., seed+1], ...]
            return sum(seeds, [])

        envs = self.envs
        rollout_cache = [{"env_id": entry['env_id'], "history": [], "group_id": entry['group_id'], "tag": entry['tag'], "penalty": 0} for entry in envs]

        # reset all environments
        # if self.mode == "train":
        #     seed = random.randint(0, 1000000) if seed is None else seed # get a random seed
            
        # else:
        #     seed = 123
        # reset all environments
        if seed is None:
            if self.mode == "train":
                if self.base_seed is not None:
                    seed = self.base_seed + self.seed_counter
                    self.seed_counter += self.env_groups
                else:
                    seed = random.randint(0, 1000000)
            else:
                seed = 123 if self.base_seed is None else self.base_seed
        else:
            if self.mode == "train" and self.base_seed is not None:
                self.seed_counter = seed - self.base_seed + 1
        # import pdb; pdb.set_trace()
        seeds = _expand_seed(seed)
        for seed, entry in zip(seeds, envs):
            entry['env'].reset(seed=seed)
            entry['status'] = EnvStatus(seed=seed)

        # update rollout cache
        for cache, env in zip(rollout_cache, envs):
            next_state = self._handle_mm_state(env['env'].render())
            cache['history'] = self._update_cache_history(cache['history'], next_state=next_state, actions_left=env['max_actions_per_traj'], num_actions_info=None)
            
        self.rollout_cache = rollout_cache
        return rollout_cache

    def step(self, all_env_inputs: List[Dict]):
        """Step the environments.
        1. extract valid actions from the action lookup table (if exists) and execute the actions, and update rollout cache
        2. Since rollout does not need to act over done envs, whenever the environment is done, we only update rollout cache, but not output env_outputs.
        Input:
        all_env_inputs: List[Dict]
            {env_id: int, llm_response: str, actions: List[str]}
            NOTE: should use env_id as index for existing some already done envs
        env_outputs: List[Dict]
            {env_id: int, history: List[Dict][{state: str, actions: List[str], reward: float, info: Dict, llm_response: str, llm_raw_response: str, (Optional)images: List[PIL.Image.Image]}]}
        """
        def _execute_actions(env, actions):
            acc_reward, turn_info, turn_done = 0, {}, False
            executed_actions = []
            for action in actions:
                obs, reward, done, info = env.step(action)
                # print(f"action: {action}, obs: {obs}, reward: {reward}, done: {done}, info: {info}")
                acc_reward += reward
                turn_info.update(info) # NOTE: currently use last info for multi-action
                executed_actions.append(action)
                if done:
                    turn_done = True
                    break
            return acc_reward, turn_info, turn_done, executed_actions

        def _log_env_state(status, history, cur_obs, max_actions_per_traj, executed_actions, all_actions, acc_reward, turn_done, turn_info, env_input):
            obs = self._handle_mm_state(cur_obs)
            status.num_actions += len(executed_actions)
            status.rewards.append(acc_reward) # NOTE use turn-wise acc_reward
            actions_left = max_actions_per_traj - status.num_actions
            if turn_done:
                status.terminated = True # TODO check terminated definition in gymnasium
                status.truncated = not turn_info.get('success', False)
            history = self._update_cache_history(history, next_state=obs, actions_left=actions_left, num_actions_info={
                'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
                'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response']
            })
            # filter out invalid actions
            # history = [content for content in history[:-1] if content['actions']] + [history[-1]]
            return status, history

        envs = self.envs
        env_outputs = []

        for env_input in all_env_inputs:
            acc_reward, turn_info, turn_done = 0, {}, False
            entry = envs[env_input['env_id']]
            env_id, env = entry['env_id'], entry['env']
            actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions

            # execute actions in envs
            valid_actions = self._extract_map_valid_actions(entry, env_input['actions'])
            acc_reward, turn_info, turn_done, executed_actions = _execute_actions(env, valid_actions[:actions_left_before])
            # print(f"acc_reward, turn_info, turn_done, executed_actions: {acc_reward}, {turn_info}, {turn_done}, {executed_actions}")
            if len(valid_actions) != len(env_input['actions']) or not valid_actions:
                self.rollout_cache[env_id]["penalty"] += self.sys_config.es_manager.format_penalty
                
            status, history = _log_env_state(entry['status'], self.rollout_cache[env_id]['history'], entry['env'].render(), entry['max_actions_per_traj'], executed_actions, valid_actions, acc_reward, turn_done, turn_info, env_input)
            entry['status'] = status
            if entry['status'].num_actions >= entry['max_actions_per_traj'] and not turn_done:
                entry['status'].truncated = True
                entry['status'].terminated = True
                turn_done = True
            self.rollout_cache[env_id]['history'] = history
            if not turn_done: # NOTE done environments are not sent for further llm generation (for efficiency)
                env_outputs.append(self.rollout_cache[env_id])
        # print(all_env_inputs)
        # print(entry['status'].num_actions)
        # print([env["status"].num_actions for env in envs])
        # calcuate terminated but not truncated
        # print([env["status"].terminated and (not env["status"].truncated) for env in envs])
        # print('success_ratio: ', np.mean([env["status"].terminated and (not env["status"].truncated) for env in envs]))
        # Calculate pass@k metrics using reshape
        success = np.array([float(env["status"].terminated and (not env["status"].truncated)) for env in envs])
        success = success.reshape(self.env_groups, self.group_size)
        
        
        # # Calculate and print pass rates
        # full_pass = np.mean(np.any(success, axis=1))
        # print(f"Full pass rate: {full_pass:.3f}")
        
        # for k in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        #     sampled = np.array([np.random.choice(group, size=k, replace=False) for group in success])
        #     pass_k = np.mean(np.any(sampled, axis=1))
        #     print(f"Pass@{k} rate: {pass_k:.3f}")

        # # breakpoint()
        # pdb.set_trace()
        
        return env_outputs

    def get_rollout_states_add_worldmodel(self):
        """Get the final output for all environment"""
        envs = self.envs
        rollout_cache = self.rollout_cache
        all_states = []
        max_len = 0
        # add metrics to rollout cache
        for entry, cache in zip(envs, rollout_cache):
            status = entry['status']
            env_metric = {
                'success': float(status.terminated and (not status.truncated)),
                'num_actions': status.num_actions,
            }
            this_states = []
            custom_metric = {}
            for turn in cache['history']:
                this_states.append(turn['state'])
                for k, v in turn.get('info', {}).items():
                    if k == 'success':
                        continue
                    if k not in custom_metric:
                        custom_metric[k] = []
                    custom_metric[k].append(float(v))
            for k, v in custom_metric.items():
                env_metric[k] = np.sum(v) / (len(cache['history']) - 1) # NOTE: exclude the last observation

            cache['history'][-1]['metrics'] = custom_metric
            env_metric = {f"{entry['tag']}/{k}": v for k, v in env_metric.items()}
            cache['metrics'] = env_metric
            all_states.append(this_states)
            max_len = max(max_len, len(this_states))
        
        # Pad all states to the same length
        padded_states = []
        for states in all_states:
            if len(states) < max_len:
                # Pad with the last state
                states = states + [states[-1]] * (max_len - len(states))
            padded_states.append(states)
            
        # Compute pass@k (group-level success) where a group is successful if any env in the group succeeds
        group_success = {}
        
        for entry in envs:
            gid = entry['group_id']
            env_success = float(entry['status'].terminated and (not entry['status'].truncated))
            if gid not in group_success:
                group_success[gid] = env_success
                
            else:
                group_success[gid] = max(group_success[gid], env_success)
        

        group_success_counts: Dict[int, float] = {}   # successful envs count per group
        group_total_counts: Dict[int, int] = {}       # total envs count per group

        # Aggregate successes and totals for every group
        for entry in envs:
            gid = entry['group_id']
            success_flag = float(entry['status'].terminated and (not entry['status'].truncated))
            group_success_counts[gid] = group_success_counts.get(gid, 0.0) + success_flag
            group_total_counts[gid] = group_total_counts.get(gid, 0) + 1

        # Calculate per-group mean success (success rate)
        group_success_mean = {
            gid: group_success_counts[gid] / group_total_counts[gid]
            for gid in group_total_counts
        }

        # Overall pass_mean_k is the average of per-group success rates
        mean_k = float(np.mean(list(group_success_mean.values()))) if group_success_mean else 0.0

        # Record per-env metrics for success_mean_k / success_meank
        for entry, cache in zip(envs, rollout_cache):
            tag = entry['tag']
            gid = entry['group_id']
            cache['metrics'][f"{tag}/success_mean_k"] = float(group_success_mean[gid])
            # cache['metrics'][f"{tag}/success_meank"]  = float(group_success_mean[gid])
        
        
        # Override the stored success metric with group-level success
        for entry, cache in zip(envs, rollout_cache):
            tag = entry['tag']
            cache['metrics'][f"{tag}/success_passk"] = float(group_success[entry['group_id']])
        
        return rollout_cache, np.array(padded_states)
    
    def get_rollout_states_base(self):
        """Get the final output for all environment"""
        envs = self.envs
        rollout_cache = self.rollout_cache

        # add metrics to rollout cache
        for entry, cache in zip(envs, rollout_cache):
            status = entry['status']
            env_metric = {
                'success': float(status.terminated and (not status.truncated)),
                'num_actions': status.num_actions,
            }
            custom_metric = {}
            for turn in cache['history']:
                for k, v in turn.get('info', {}).items():
                    # print(f"k: {k}, v: {v}")
                    # pdb.set_trace()
                    if k == 'success':
                        continue
                    if k not in custom_metric:
                        custom_metric[k] = []
                   
                    custom_metric[k].append(float(v))
            for k, v in custom_metric.items():
                env_metric[k] = np.sum(v) / (len(cache['history']) - 1) # NOTE: exclude the last observation

            cache['history'][-1]['metrics'] = custom_metric
            env_metric = {f"{entry['tag']}/{k}": v for k, v in env_metric.items()}
            cache['metrics'] = env_metric

        # Compute pass@k (group-level success) where a group is successful if any env in the group succeeds
        group_success = {}
        
        for entry in envs:
            gid = entry['group_id']
            env_success = float(entry['status'].terminated and (not entry['status'].truncated))
            if gid not in group_success:
                group_success[gid] = env_success
                
            else:
                group_success[gid] = max(group_success[gid], env_success)
        

        group_success_counts: Dict[int, float] = {}   # successful envs count per group
        group_total_counts: Dict[int, int] = {}       # total envs count per group

        # Aggregate successes and totals for every group
        for entry in envs:
            gid = entry['group_id']
            success_flag = float(entry['status'].terminated and (not entry['status'].truncated))
            group_success_counts[gid] = group_success_counts.get(gid, 0.0) + success_flag
            group_total_counts[gid] = group_total_counts.get(gid, 0) + 1

        # Calculate per-group mean success (success rate)
        group_success_mean = {
            gid: group_success_counts[gid] / group_total_counts[gid]
            for gid in group_total_counts
        }

        # Overall pass_mean_k is the average of per-group success rates
        mean_k = float(np.mean(list(group_success_mean.values()))) if group_success_mean else 0.0

        # Record per-env metrics for success_mean_k / success_meank
        for entry, cache in zip(envs, rollout_cache):
            tag = entry['tag']
            gid = entry['group_id']
            cache['metrics'][f"{tag}/success_mean_k"] = float(group_success_mean[gid])
            # cache['metrics'][f"{tag}/success_meank"]  = float(group_success_mean[gid])
        
        
        # Override the stored success metric with group-level success
        for entry, cache in zip(envs, rollout_cache):
            tag = entry['tag']
            cache['metrics'][f"{tag}/success_passk"] = float(group_success[entry['group_id']])
        return rollout_cache


    def _update_cache_history(self, history: List[Dict], next_state, actions_left, num_actions_info: Optional[Dict] = None):
        """
        Update last step info and append state to history
        """
        if num_actions_info is not None: # update last step info
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)
        
        entry = {} # append state to history
        if isinstance(next_state, str): # text state
            entry['state'] = next_state
        else: # multimodal state
            entry['state'] = "<images>" * len(next_state)
            entry['images'] = next_state
        entry['actions_left'] = actions_left
        history.append(entry)
        return history

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        """extract valid actions from the action lookup table (if exists)"""
        mapped_actions = []
        action_lookup = getattr(entry['env'].config, 'action_lookup', None)
        if action_lookup is None:
            mapped_actions = actions
        else: # the envs have pre-defined action lookup
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            actions = [action.lower() for action in actions]
            mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]
            def match_template(action: str, templates: dict) -> bool:
                action_tokens = action.split()
                action_tokens = [token for token in action_tokens if not token.isdigit()] # remove number in action
                for template in templates:
                    template_tokens = template.split()

                    if len(action_tokens) != len(template_tokens):
                        continue

                    match = True
                    for atok, ttok in zip(action_tokens, template_tokens):
                        if ttok.startswith('<') and ttok.endswith('>'):
                            continue  
                        if atok != ttok:
                            match = False
                            break

                    if match:
                        return True

                return False

            if 'look' in rev_action_lookup.keys():
                mapped_actions = [action for action in actions if match_template(action, rev_action_lookup)]
        return mapped_actions
    
    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray]]):
        """Handle the state from the environment
        """
        if isinstance(state, str): # text state
            return state
        elif isinstance(state, np.ndarray): # when env state is a single image, convert it to a list to unify output format
            state = [state]
        results = [PIL.Image.fromarray(_state, mode='RGB') for _state in state]
        return results
        
    def render(self):
        rendered_list = [entry['env'].render() for entry in self.envs]
        return rendered_list

    def close(self):
        for entry in self.envs:
            entry['env'].close()




@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    """
    Unit test for EnvStateManager
    """
    es_manager = EnvStateManager(config, mode="train")
    print("Initializing environments...")
    es_manager.reset(seed=123)

    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRunning step for training environments...")
    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")

    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go left, go up",
            "llm_response": "Go left, go up",
            "actions": ["left", "up"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go up, go up",
            "llm_response": "Go up, go up",
            "actions": ["up", "up", "up", "up", "up"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRendering final output...")
    final_outputs = es_manager.get_rollout_states()
    print(f"final outputs[:4]: {final_outputs[:4]}")
    
    print("\nClosing environments...")
    es_manager.close()
    print("Test completed successfully!")


if __name__ == "__main__":
	main()
