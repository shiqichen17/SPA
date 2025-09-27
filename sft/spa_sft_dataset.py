# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""
import pdb

from typing import List, Union

import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer

import numpy as np

from typing import Union
import numpy as np
import copy

def apply_chat_template(
        tokenizer, 
        messages: Union[np.ndarray, str], 
        response: str,
        with_thinking=True, 
        **kwargs
    ):
    """
    Apply a chat template to given messages.
    ================================
    Args:
        - tokenizer: The tokenizer to use.
        - messages (Union[np.ndarray, str]): The messages to apply the template to.
            - str: The messages are already formatted.
            - np.ndarray: The messages are not formatted.
        - with_thinking: Whether the assistant needs to output think tags.
            - e.g., <|im_start|>assistant\n --> <|im_start|>assistant\n<think>, assistant needs to output </think> tags.
            - assistant message is added <think></think> at the beginning
        - kwargs: Additional keyword arguments to pass to the tokenizer.apply_chat_template method.
    ================================
    Returns:
        The formatted messages (str).
    """
    if isinstance(messages, str):
        return messages
    # pdb.set_trace()
    messages=np.array(messages)
    # print(len(messages))
    # pdb.set_trace()
    assert isinstance(messages, np.ndarray), "The messages must be a numpy array."
    messages = copy.deepcopy(messages.tolist())
    assert messages[-1]['role'] == 'user', "The last message must be a user message."
    if not with_thinking:
        prompt_chat_str = tokenizer.apply_chat_template(messages, **kwargs)
    if with_thinking:
        for msg in messages:
            if msg['role'] == 'assistant':
                msg['content'] = f"<think></think>{msg['content']}"
        prompt_chat_str = tokenizer.apply_chat_template(messages, **kwargs)
        prompt_chat_str = f"{prompt_chat_str}<think>"
        response_chat_str = f"</think>{response}"
    return prompt_chat_str, response_chat_str
class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 prompt_key='prompt',
                 prompt_dict_keys=None,
                 response_key='response',
                 response_dict_keys=None,
                 max_length=1024,
                 truncation='left'):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        ########################## Original Codes ########################## 
        # self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        # self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        ########################## Original Codes ##########################
        assert isinstance(prompt_key, str)
        assert isinstance(response_key, str)
        self.prompt_key = prompt_key
        self.response_key = response_key

        self.prompt_dict_keys = [] if not prompt_dict_keys else prompt_dict_keys
        self.response_dict_keys = [] if not response_dict_keys else response_dict_keys

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_tokenize(self):

        def series_to_item(ls):
            import pandas, numpy
            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        self.prompt_dict_keys=[]
        self.response_dict_keys=[]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print(f'self.prompts={self.prompts}')
                raise
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print(f'self.responses={self.responses}')
                raise
        self.responses = self.responses.tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # Preprocess text to replace observation and prediction tags with special tokens
        def preprocess_text_for_tokenization(text):
            """Preprocess text to replace observation and prediction tags with special tokens before tokenization."""
            import re
            
            # Replace observation tags and their content with special token
            text = re.sub(r'<observation>.*?</observation>', '<obs_content>', text, flags=re.DOTALL)
            
            # Replace prediction tags and their content with special token
            text = re.sub(r'<prediction>.*?</prediction>', '<pred_content>', text, flags=re.DOTALL)
            
            return text

        # Apply preprocessing to prompt and response
        if isinstance(prompt, str):
            prompt = preprocess_text_for_tokenization(prompt)
        elif isinstance(prompt, np.ndarray):
            # If prompt is an array of messages, process each message
            prompt = copy.deepcopy(prompt.tolist())
            for msg in prompt:
                if isinstance(msg, dict) and 'content' in msg:
                    msg['content'] = preprocess_text_for_tokenization(msg['content'])
        
        response = preprocess_text_for_tokenization(response)

        ########################## Original Codes ##########################
        # # apply chat template
        # prompt_chat = [{'role': 'user', 'content': prompt}]

        # # string
        # prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        # response_chat_str = response + tokenizer.eos_token
        ########################## Original Codes ##########################
        prompt_chat_str, response_chat_str = apply_chat_template(tokenizer, prompt, response, with_thinking=True, add_generation_prompt=True, tokenize=False)
        response_chat_str = response_chat_str + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors='pt', add_special_tokens=False)
        prompt_ids = prompt_ids_output['input_ids'][0]
        prompt_attention_mask = prompt_ids_output['attention_mask'][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors='pt', add_special_tokens=False)
        response_ids = response_ids_output['input_ids'][0]
        response_attention_mask = response_ids_output['attention_mask'][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }