
import torch
import warnings
warnings.filterwarnings("ignore")
import os
import json
  
from openai import OpenAI



DEFAULT_DATASET_PATH = "/root/gpufree-data/sim_vla/SimVLA/datasets/metas/"

# 可选：设置随机种子以获得可重复的结果
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_libero_dataset_task_prompt(dataset_path=DEFAULT_DATASET_PATH):

    res = {}
    tasks = os.listdir(dataset_path)
    for task in tasks:
        if task not in res:
            res[task] = []
        tmp_path = os.path.join(dataset_path, task)
        demo_files = sorted(os.listdir(tmp_path))

        for file in demo_files:
            print(file)
            if "SCENE" in file:
                raw_file = file.split("SCENE")[-1].split("_")[1:-1]
            else:
                raw_file = file.split("SCENE")[-1].split("_")[0:-1]
            prompt = "_".join(raw_file)
            res[task].append(prompt)
    
    return res


def use_llm_generate_simi_text(task_desc, api_key):

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com")


    prompt = f"{task_desc}; 生成五句和这句话意思相同，表达不同的英文句子"
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    res = response.choices[0].message.content

    return res


if __name__ == "__main__": 


    set_seed(42)
    save_root = "/root/workspace/sim_vla/SimVLA/tools/noise_dataset/libero_prompts_noise"
    os.makedirs(save_root, exist_ok=True)
    api_key = None
    assert api_key is not None, "please offer llm api key"

    libero_task_prompts = _get_libero_dataset_task_prompt()

    for task, task_prompts in libero_task_prompts.items():
        print(f"processing {task}")
        task_res = {}
        for task_prompt in task_prompts:
            print(task_prompt)
            simi_prompts = use_llm_generate_simi_text(task_prompt, api_key)
            simi_prompts_ = []
            simi_prompts = simi_prompts.split("\n")
            for line in simi_prompts:
                line = line.split(". ")[1].strip()
                print(line)
                line = line.replace(",", "").replace(".","")

                simi_prompts_.append(line)
            task_res[task_prompt] = simi_prompts_
       
        path = f"{save_root}/{task}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(task_res, f, indent=4, ensure_ascii=False)
            

