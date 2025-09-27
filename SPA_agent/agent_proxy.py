from .ctx_manager import ContextManager
from .es_manager import EnvStateManager
from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from .base_llm import ConcurrentLLM
import pdb
import sys
import os
import logging
import random
import re
from datetime import datetime


class VllmWrapperWg: # Thi is a developing class for eval and test
	def __init__(self, config, tokenizer):
		self.config = config
		self.tokenizer = tokenizer
		model_name = config.actor_rollout_ref.model.path
		ro_config = config.actor_rollout_ref.rollout

		print(f"[DEBUG] loading model from: {model_name}")
		print(f"[DEBUG] transformers cache: {os.environ.get('TRANSFORMERS_CACHE')}")
		print(f"[DEBUG] vllm cache dir: {os.environ.get('VLLM_CACHE_DIR')}")

		self.llm = LLM(
			model_name,
            enable_sleep_mode=True,
            tensor_parallel_size=ro_config.tensor_model_parallel_size,
            dtype=ro_config.dtype,
            enforce_eager=ro_config.enforce_eager,
            gpu_memory_utilization=ro_config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=ro_config.max_model_len,
            disable_log_stats=ro_config.disable_log_stats,
            max_num_batched_tokens=ro_config.max_num_batched_tokens,
            enable_chunked_prefill=ro_config.enable_chunked_prefill,
            enable_prefix_caching=True,
		)
		print("LLM initialized")
		self.sampling_params = SamplingParams(
			max_tokens=ro_config.response_length,
			temperature=ro_config.val_kwargs.temperature,
			top_p=ro_config.val_kwargs.top_p,
			top_k=ro_config.val_kwargs.top_k,
			# min_p=0.1,
		)

	def generate_sequences(self, lm_inputs: DataProto):
		"""
		Convert the input ids to text, and then generate the sequences. Finally create a dataproto. 
		This aligns with the verl Worker Group interface.
		"""
		# NOTE: free_cache_engine is not used in the vllm wrapper. Only used in the verl vllm.
		# cache_action = lm_inputs.meta_info.get('cache_action', None)

		input_ids = lm_inputs.batch['input_ids']
		input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
		input_texts = [i.replace("<|endoftext|>", "") for i in input_texts]

		outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
		texts = [output.outputs[0].text for output in outputs] 
		lm_outputs = DataProto()
		lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
		lm_outputs.meta_info = lm_inputs.meta_info
		
		return lm_outputs
	
class ApiCallingWrapperWg:
    """Wrapper class for API-based LLM calls that fits into the VERL framework"""
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs
        
        
        self.llm = ConcurrentLLM(
			provider=model_info.provider_name,
            model_name=model_info.model_name,
            max_concurrency=config.model_config.max_concurrency
        )
        
        print(f'API-based LLM ({model_info.provider_name} - {model_info.model_name}) initialized')


    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        """
        Convert the input ids to text, make API calls to generate responses, 
        and create a DataProto with the results.
        """

        messages_list = lm_inputs.non_tensor_batch['messages_list'].tolist()
        results, failed_messages = self.llm.run_batch(
            messages_list=messages_list,
            **self.llm_kwargs
        )
        assert not failed_messages, f"Failed to generate responses for the following messages: {failed_messages}"

        texts = [result["response"] for result in results]
        print(f'[DEBUG] texts: {texts}')
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
        lm_outputs.meta_info = lm_inputs.meta_info
        
        return lm_outputs

class RandomActionWrapperWg:
    """Wrapper class for random action generation that fits into the VERL framework"""
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        print(f'Random Action Wrapper initialized')

    def generate_random_action(self):
        """Generate three random actions from up, down, left, right, separated by ||."""
        actions = ["up", "down", "left", "right"]
        three_actions = [random.choice(actions) for _ in range(3)]
        return " || ".join(three_actions)

    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        """
        Generate random action responses following the strict format:
        <think> <observation>...</observation> I will push the box to the target. <prediction>...</prediction></think><answer>ACTION</answer>
        """
        messages_list = lm_inputs.non_tensor_batch['messages_list'].tolist()
        
        # Generate random actions for each message
        response_texts = []
        for messages in messages_list:
            # Find the last assistant message to modify
            assistant_message = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    assistant_message = msg
                    break
            
            if assistant_message:
                # Generate random action
                random_action = self.generate_random_action()
                
                # Create the response following the strict format
                # Extract observation and prediction from the original message
                content = assistant_message.get("content", "")
                
                # Extract observation
                obs_match = re.search(r'<observation>(.*?)</observation>', content, re.DOTALL)
                observation = obs_match.group(1) if obs_match else ""
                
                # Extract prediction  
                pred_match = re.search(r'<prediction>(.*?)</prediction>', content, re.DOTALL)
                prediction = pred_match.group(1) if pred_match else ""
                
                # Create new response with random action
                response = f"<think> <observation>{observation}</observation> I will push the box to the target. <prediction>{prediction}</prediction></think><answer>{random_action}</answer>"
                response_texts.append(response)
            else:
                # Fallback if no assistant message found
                random_action = self.generate_random_action()
                response = f"<think> <observation></observation> I will push the box to the target. <prediction></prediction></think><answer>{random_action}</answer>"
                response_texts.append(response)
        
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
            'response_texts': response_texts,
            'env_ids': lm_inputs.non_tensor_batch['env_ids'],
            'group_ids': lm_inputs.non_tensor_batch['group_ids']
        }
        lm_outputs.meta_info = lm_inputs.meta_info
        
        return lm_outputs

class LLMAgentProxy:
	"""
	The proxy means the llm agent is trying to generate some rollout **at this time**, **at this model state**, **at this env state from the env config**
	"""
	def __init__(self, config, actor_rollout_wg, tokenizer):
		self.config = config
		self.train_ctx_manager = ContextManager(config, tokenizer, mode="train")
		self.train_es_manager = EnvStateManager(config, mode="train")
		self.val_ctx_manager = ContextManager(config, tokenizer, mode="val")
		self.val_es_manager = EnvStateManager(config, mode="val")
		self.actor_wg = actor_rollout_wg
		self.tokenizer = tokenizer
		# -------- prepare a validation trajectory file path (once per proxy) --------
		timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		# model name (strip path separators)
		model_path  = getattr(self.config, "model_path", None)
		if model_path is None and hasattr(self.config, "actor_rollout_ref"):
			model_path = getattr(self.config.actor_rollout_ref.model, "path", "unknown_model")
		model_name  = os.path.basename(str(model_path)).replace("/", "_")
		# ctx manager mode (e.g. base / add_worldmodel ...)
		mode_name   = getattr(self.config.ctx_manager, "mode", "unknown_mode")
		file_name  = f"valtraj_{model_name}_{mode_name}_{timestamp}.txt"
		out_dir    = "val_trajectories"
		os.makedirs(out_dir, exist_ok=True)
		self.valtraj_file_path = os.path.join(out_dir, file_name)

	def generate_sequences(self, lm_inputs: DataProto):
		# TODO: add kv cache both for the vllm wrapper here and for verl vllm.
		if isinstance(self.actor_wg, RayWorkerGroup):
			padded_lm_inputs, pad_size = pad_dataproto_to_divisor(lm_inputs, self.actor_wg.world_size)
			
			padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)
			# pdb.set_trace()
			lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
			lm_outputs.meta_info = lm_inputs.meta_info
			lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
		elif isinstance(self.actor_wg, VllmWrapperWg) or isinstance(self.actor_wg, ApiCallingWrapperWg) or isinstance(self.actor_wg, RandomActionWrapperWg):
			lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
		else:
			raise ValueError(f"Unsupported actor worker type: {type(self.actor_wg)}")

		return lm_outputs

	def rollout(self, dataproto: DataProto, val=False):
		es_manager = self.val_es_manager if val else self.train_es_manager
		ctx_manager = self.val_ctx_manager if val else self.train_ctx_manager
		env_outputs = es_manager.reset()
	
		for i in range(self.config.agent_proxy.max_turn):
			lm_inputs: DataProto = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
			lm_inputs.meta_info = dataproto.meta_info # TODO: setup vllm early stop when max length is reached. make sure this can be done
			lm_outputs: DataProto = self.generate_sequences(lm_inputs) #len: 128
			env_inputs: List[Dict] = ctx_manager.get_env_inputs(lm_outputs)
			env_outputs: List[Dict] = es_manager.step(env_inputs)
			
			if len(env_outputs) == 0: # all finished
				break
		
		if self.config.ctx_manager.mode=='add_worldmodel':
			rollout_states,all_states = es_manager.get_rollout_states_add_worldmodel()

		
		elif self.config.ctx_manager.mode=='base':
			rollout_states = es_manager.get_rollout_states_base()
		
		rollouts = ctx_manager.formulate_rollouts(rollout_states)
		# self.tokenizer.batch_decode(rollouts.batch['input_ids'], skip_special_tokens=False) # see all the trajectories
		messages_list = rollouts.non_tensor_batch["messages_list"]
		
		for i, (messages, rollout_state) in enumerate(zip(messages_list, rollout_states)):
			if i >= 10:
				break
			print(f"Env {i}:")
			print("Messages:")
			for msg in messages:
				print(msg)
			# print("Success:", rollout_state["metrics"].get("success", None))
			print("-" * 40)
		
		# Get success status for each environment
		# pdb.set_trace()
		# success_list = [rollout_state["metrics"].get("FrozenLake/success", False) for rollout_state in rollout_states]
		# success_list = [rollout_state["metrics"].get("SimpleSokoban/success", False) for rollout_state in rollout_states]
		success_list = [rollout_state["metrics"].get("FrozenLake/success", rollout_state["metrics"].get("SimpleSokoban/success", rollout_state["metrics"].get("O3Sokoban/success", False))) for rollout_state in rollout_states]

		# print("Success status for all environments:", success_list)
		# Calculate and print success rate
		success_rate = sum(success_list) / len(success_list)
		print(f"Overall success rate: {success_rate:.3f}")
		'''
		if val:
			# Start Generation Here
			# Save messages and corresponding rollout metrics into a JSON file
			try:
				# Dynamically import to avoid polluting global namespace
				import os, json
				# Record (best-effort) the current training step so that it can be
				# referenced later when dumping validation trajectories.
				
				# -------- use persistent save path --------
				file_path  = self.valtraj_file_path
				train_step = 0
				# -------- dump trajectories --------
				
				with open(file_path, "a", encoding="utf-8") as fp:
					train_step=os.environ.get("STEP", 0)
					fp.write(f" Current training step: {train_step}\n")
					# pdb.set_trace()
					for i, (messages, rollout_state) in enumerate(zip(messages_list, rollout_states)):
						fp.write(f"Env {i}\n")
						fp.write("Messages:\n")
						for msg in messages:
							# msg can be dict / str
							if isinstance(msg, (dict, list)):
								fp.write(json.dumps(msg, ensure_ascii=False) + "\n")
							else:
								fp.write(str(msg) + "\n")
						fp.write("Metrics:\n")
						fp.write(json.dumps(rollout_state.get("metrics", {}), ensure_ascii=False) + "\n")
						fp.write("-" * 60 + "\n")
				print(f"[Validation] Trajectories saved to {file_path}")
			except Exception as e:
				print(f"[Validation] Failed to save trajectories: {e}")
		'''
		if self.config.ctx_manager.mode=='add_worldmodel':
			return rollouts,rollout_states,all_states
		elif self.config.ctx_manager.mode=='base':
			return rollouts,rollout_states

def init_logging(to_file_only=False, log_dir="log"):
    """Set up logging: redirect stdout/stderr to file and optionally keep console output."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"debug_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

    # Clear existing handlers (important if libraries already added theirs)
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
@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):

	init_logging(to_file_only=False)  # log to file + keep terminal output
	# detect config name from python -m ragen.llm_agent.agent_proxy --config_name frozen_lake
	os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
	os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", str(config.system.CUDA_VISIBLE_DEVICES))
	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
	actor_wg = VllmWrapperWg(config, tokenizer)
	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
	import time
	start_time = time.time()
	rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample':config.actor_rollout_ref.rollout.do_sample, 'validate': True}), val=True)
	end_time = time.time()
	print(f'rollout time: {end_time - start_time} seconds')
	# print rollout rewards from the rm_scores
	rm_scores = rollouts.batch["rm_scores"]
	metrics = rollouts.meta_info["metrics"]
	avg_reward = rm_scores.sum(-1).mean().item()
	print(f'rollout rewards: {avg_reward}')
	print(f'metrics:')
	for k, v in metrics.items():
		print(f'{k}: {v}')

# @hydra.main(version_base=None, config_path="../../config", config_name="evaluate_api_llm")
# def main(config):
# 	# detect config name from python -m ragen.llm_agent.agent_proxy --config_name frozen_lake
# 	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
# 	actor_wg = ApiCallingWrapperWg(config, tokenizer)
# 	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
# 	import time
# 	start_time = time.time()
# 	rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}), val=True)
# 	print(f'[DEBUG] rollouts: {rollouts}')
# 	end_time = time.time()
# 	print(f'rollout time: {end_time - start_time} seconds')
# 	# print rollout rewards from the rm_scores
# 	rm_scores = rollouts.batch["rm_scores"]
# 	metrics = rollouts.meta_info["metrics"]
# 	avg_reward = rm_scores.sum(-1).mean().item()
# 	print(f'rollout rewards: {avg_reward}')
# 	print(f'metrics:')
# 	for k, v in metrics.items():
# 		print(f'{k}: {v}')



if __name__ == "__main__":
	main()