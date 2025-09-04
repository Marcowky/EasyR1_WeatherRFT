import re
import copy
import json
import os

from typing import Any, Dict, List


def format_reward(response: str) -> float:
    """Reward function that checks if the completion has a specific format."""
    # 计算 <think> 和 <answer> 的数量
    single_think = response.count("<think>") == 1 and response.count("</think>") == 1
    single_answer = response.count("<answer>") == 1 and response.count("</answer>") == 1

    # 判断是否有 <think> 和 <answer> 标签
    has_think = bool(re.search(r'^<think>.*?</think>', response, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>$', response, re.DOTALL))

    # 判断是否同时有 <think> 和 <answer> 标签
    has_think_answer = bool(re.search(r"^<think>.*?</think>\s*<answer>.*?</answer>$", response, re.DOTALL))
    
    # 计算分数
    score = 0.0
    if has_think and single_think:
        score += 0.25
    if has_answer and single_answer:
        score += 0.25
    if has_think_answer and single_think and single_answer:
        score += 0.5
    
    return score


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Reward function that checks if the completion matches the correct answer choice (A/B/C/D)."""
    reward = 0.0
    
    try:
        # 使用 re.findall 获取所有匹配项
        matches = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)

        # 检查是否有匹配项
        if matches:
            # 取最后一个匹配项
            content_answer = matches[-1].strip()
        else:
            sentences = re.split(r'[。！？.!?]', response)
            # 去除空的句子
            sentences = [s.strip() for s in sentences if s.strip()]
            content_answer = sentences[-1] if sentences else ""
        
        # 正则表达式提取出 A/B/C/D 中的一个
        student_answer_match = re.search(r'[A-D]', content_answer)
        student_answer = student_answer_match.group(0) if student_answer_match else ""

        # 将 ground_truth 和 student_answer 转换为大写
        ground_truth = ground_truth.upper().strip()
        student_answer = student_answer.upper().strip()
        
        # 判断对错
        if student_answer == ground_truth:
            # 若 content_answer 不做任何提取与处理，就已经是 A/B/C/D 中的一个，则直接奖励 1.0
            if content_answer == student_answer:
                reward = 1.0
            else:
                reward = 0.5
                
    except Exception:
        pass
        
    return reward


def compute_score(reward_inputs: List[Dict[str, Any]], cur_stat: str, cur_step: int, save_path: str, format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    save_datas = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
        # 保存详细的评分数据
        temp_data = copy.deepcopy(reward_input)
        del temp_data["response_length"]
        temp_data['score'] = scores[-1]
        save_datas.append(temp_data)

    # 构造保存路径
    base_save_path = os.path.join(save_path, 'detailed_scores', f"stat_{cur_stat}_step_{cur_step}")
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(base_save_path), exist_ok=True)
    # 若文件已存在，则重命名备份
    counter = 0
    json_save_path = f"{base_save_path}_{counter}.json"
    while os.path.exists(json_save_path):
        counter += 1
        json_save_path = f"{base_save_path}_{counter}.json"
    # 保存到 json
    with open(json_save_path, "w") as f:
        json.dump(save_datas, f, indent=4, ensure_ascii=False)

    return scores
