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

import importlib.util
import inspect
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Optional, Tuple, TypedDict, Union

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict, total=False):
    response: str
    response_length: int
    ground_truth: str
    problem: Any
    multi_modal_data: Any


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


def _maybe_add_non_tensor_field(reward_input: RewardInput, data: DataProto, key: str, index: int) -> None:
    values = data.non_tensor_batch.get(key)
    if values is not None:
        reward_input[key] = values[index]


def _reward_accepts_context(reward_fn: Callable) -> bool:
    try:
        signature = inspect.signature(reward_fn)
    except (TypeError, ValueError):
        return False

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters.values()):
        return True

    return {"cur_stat", "cur_step", "save_path"}.issubset(parameters)


def _call_reward_fn(
    reward_fn: Callable,
    reward_inputs: Union[RewardInput, list[RewardInput]],
    cur_stat: str,
    cur_step: int,
    save_path: str,
) -> Union[RewardScore, list[RewardScore]]:
    if _reward_accepts_context(reward_fn):
        return reward_fn(reward_inputs, cur_stat=cur_stat, cur_step=cur_step, save_path=save_path)

    return reward_fn(reward_inputs)


class SequentialFunctionRewardManagerMixin:
    reward_fn: SequentialRewardFunction

    def compute_reward_sequential(
        self, data: DataProto, cur_stat: str, cur_step: int, save_path: str
    ) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            reward_input: RewardInput = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            _maybe_add_non_tensor_field(reward_input, data, "problem", i)
            _maybe_add_non_tensor_field(reward_input, data, "multi_modal_data", i)
            score = _call_reward_fn(
                self.reward_fn,
                reward_input,
                cur_stat,
                cur_step,
                save_path,
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManagerMixin:
    reward_fn: BatchRewardFunction

    def compute_reward_batch(
        self, data: DataProto, cur_stat: str, cur_step: int, save_path: str
    ) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            reward_input: RewardInput = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            _maybe_add_non_tensor_field(reward_input, data, "problem", i)
            _maybe_add_non_tensor_field(reward_input, data, "multi_modal_data", i)
            reward_inputs.append(reward_input)

        scores = _call_reward_fn(self.reward_fn, reward_inputs, cur_stat, cur_step, save_path)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class AutoRewardManager(BatchFunctionRewardManagerMixin, SequentialFunctionRewardManagerMixin):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        reward_name = getattr(module, "REWARD_NAME", "unknown")
        reward_type = getattr(module, "REWARD_TYPE", "batch")
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_type = reward_type
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(
        self, data: DataProto, cur_stat: str, cur_step: int, save_path: str
    ) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        if self.reward_type == "batch":
            return self.compute_reward_batch(data, cur_stat, cur_step, save_path)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(data, cur_stat, cur_step, save_path)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")
