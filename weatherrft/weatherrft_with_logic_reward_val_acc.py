import re
import copy
import json
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List


openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:50907/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
client_model = "/home/kaiyu/Model/openai/gpt-oss-20b/"

def call_client_model(query: str) -> str:
    chat_response = client.chat.completions.create(
        model=client_model,
        messages=[{
            "role": "user", 
            "content": query
        }],
        temperature=0,
        max_tokens=512
    )
    return chat_response.choices[0].message.content


def get_option_letter(content_answer: str) -> str:
    # 正则表达式提取出 A-Z 中的一个
    student_answer_match = re.search(r'[A-Z]', content_answer.upper())
    student_answer = student_answer_match.group(0) if student_answer_match else ""
    return student_answer.upper().strip()


def get_think(output: str) -> str:
    # 使用 re.findall 获取所有匹配项
    matches = re.findall(r'<think>(.*?)</think>', output, re.DOTALL)

    # 检查是否有匹配项
    if matches:
        # 取最后一个匹配项
        think_part = matches[-1].strip()
    else:
        think_part = output.strip()
    
    return think_part


def get_answer(output: str) -> str:
    # 使用 re.findall 获取所有匹配项
    matches = re.findall(r'<answer>(.*?)</answer>', output, re.DOTALL)

    # 检查是否有匹配项
    if matches:
        # 取最后一个匹配项
        content_answer = matches[-1].strip()
    else:
        sentences = re.split(r'[。！？.!?]', output)
        # 去除空的句子
        sentences = [s.strip() for s in sentences if s.strip()]
        content_answer = sentences[-1] if sentences else ""

    return content_answer


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
        content_answer = get_answer(response)
        # 正则表达式提取出 A/B/C/D 中的一个
        student_answer = get_option_letter(content_answer)

        # 将 ground_truth 转换为大写
        ground_truth = ground_truth.upper().strip()
        
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


def logic_reward(problem: str, response: str) -> float:
    LOGIC_PROMPT = """Your task is to select the option that corresponds to the given response.
Directly output the uppercase letter of the selected option. If the response does not correspond to any of the options, output "Cannot be determined".
[Input]: {output}
[Output]: """

    reward = 0.0
    think_answer = ""

    try:
        q_and_a = problem.replace("<image>", "") + '\nResponse:\n' + get_think(response)
        query = LOGIC_PROMPT.format(output=q_and_a)
        think_answer = call_client_model(query)

        if "Cannot be determined".lower() not in think_answer.lower():
            think_answer = get_option_letter(think_answer)
        student_answer = get_option_letter(get_answer(response))

        if think_answer == student_answer:
            reward = 1.0

    except Exception:
        pass
    
    return reward, think_answer


def _process_single_reward_input(reward_input: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to process a single reward input."""
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
    
    format_score = format_reward(response)
    if format_score != 1.0:
        logic_score = 0.0
        think_answer = "Format error"
    else:
        logic_score, think_answer = logic_reward(reward_input["problem"], response)
    accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
    
    return {
        "reward_input": reward_input,
        "scores": {
            "format": format_score,
            "logic": logic_score,
            "accuracy": accuracy_score
        },
        "think_answer": think_answer
    }


def compute_score(reward_inputs: List[Dict[str, Any]], cur_stat: str, cur_step: int, save_path: str) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    reward_weights = {
        "format": 0.1,
        "logic": 0.3,
        "accuracy": 0.6
    }

    # 使用 ThreadPoolExecutor 并行处理
    with ThreadPoolExecutor(max_workers=256) as executor:
        results = list(executor.map(_process_single_reward_input, reward_inputs))

    scores = []
    save_datas = []
    
    for result in results:
        cur_result = copy.deepcopy(result)
        del cur_result["reward_input"]["response_length"]

        # 计算综合得分
        overall_score = sum(cur_result['scores'][reward] * weight for reward, weight in reward_weights.items())
        cur_result['scores']["overall"] = overall_score if cur_stat != 'val' else cur_result['scores']["accuracy"]

        # 保存详细的 reward 数据
        save_datas.append(cur_result)
        # 保存得分
        scores.append(cur_result["scores"])

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
