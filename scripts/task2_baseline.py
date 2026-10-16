import json  
import os  
import random  
import numpy as np  
import torch  
from huggingface_hub import hf_hub_download  
from transformers import AutoTokenizer, AutoModelForCausalLM 

SEED = 42 
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"  
NUM_EVAL_PROMPTS = 300  
SUBSET_PATH = "eval_sets/alpaca_eval_300_seed42.json"  
MAX_PROMPT_TOKENS = 256  
MAX_NEW_TOKENS = 256  
BATCH_SIZE = 8  
REPETITION_PENALTY = 1.0 
PROMPT_START = "Instruction:\n" 
PROMPT_END = "\n\nResponse:\n"  

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  

def make_prompt(instruction):
    return PROMPT_START + instruction + PROMPT_END
    
def save_json(data, path):
    temp_path = path + ".tmp"  
    with open(temp_path, "w", encoding="utf-8") as f:  
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)

def build_alpaca_subset():
    print("loading alpaca eval...")
    file_path = hf_hub_download(repo_id="tatsu-lab/alpaca_eval",filename="alpaca_eval.json",repo_type="dataset")
    with open(file_path, "r", encoding="utf-8") as f: 
        all_rows = json.load(f)  
    total = len(all_rows)  
    print("total questions:", total)
    if total < NUM_EVAL_PROMPTS:  
        print("not enough questions")
        return
    rng = random.Random(SEED)  
    positions = list(range(total))  
    chosen = rng.sample(positions, NUM_EVAL_PROMPTS)  
    subset = []  
    for pos in chosen:
        row = all_rows[pos]  
        entry = {} 
        entry["id"] = "alpaca_eval_" + str(pos)  
        entry["instruction"] = row["instruction"]  
        entry["dataset"] = "alpaca_eval"  
        entry["category"] = row["dataset"]
        subset.append(entry) 

    os.makedirs(os.path.dirname(SUBSET_PATH), exist_ok=True)  
    with open(SUBSET_PATH, "w", encoding="utf-8") as f:  
        json.dump(subset, f, indent=2, ensure_ascii=False)  
    print("saved", len(subset), "questions to", SUBSET_PATH)

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)  
    tokenizer.padding_side = "left"  
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:  
        tokenizer.pad_token = tokenizer.eos_token  
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16) 
    model.config.use_cache = True
    model.to("cuda:0")  
    model.eval()  
    return model, tokenizer
    
def generate_answers(model, tokenizer, model_path, prompts_path, output_path,generator_name, dataset_name="alpaca_eval", limit=None):
    set_seed(SEED)  
    with open(prompts_path, "r", encoding="utf-8") as f:  
        prompts = json.load(f)
    if limit is not None:  
        prompts = prompts[:limit]
    print("questions to answer:", len(prompts))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  
    results = [] 
    if os.path.exists(output_path):  
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)  
    done_ids = []  
    for r in results:
        done_ids.append(r["id"])
    todo = [] 
    for p in prompts:
        if p["id"] not in done_ids:
            todo.append(p)
    print("already answered:", len(done_ids), "| still to answer:", len(todo))
    config = {  
        "model_path": model_path,
        "generator_name": generator_name,
        "seed": SEED,
        "decoding": "greedy",
        "max_new_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": REPETITION_PENALTY,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "prompt_start": PROMPT_START,
        "prompt_end": PROMPT_END,
    }
    save_json(config, output_path + ".config.json")  
    for start in range(0, len(todo), BATCH_SIZE):  
        batch = todo[start:start + BATCH_SIZE]  
        texts = []  
        for item in batch:
            texts.append(make_prompt(item["instruction"]))
        enc = tokenizer(texts, return_tensors="pt", padding=True,truncation=True, max_length=MAX_PROMPT_TOKENS)  
        enc = enc.to(model.device) 
        with torch.no_grad():  
            output_ids = model.generate(**enc,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,repetition_penalty=REPETITION_PENALTY,pad_token_id=tokenizer.pad_token_id,)
        prompt_length = enc["input_ids"].shape[1]  

        for i in range(len(batch)): 
            new_tokens = output_ids[i][prompt_length:] 
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True)  
            entry = {}  
            entry["id"] = batch[i]["id"]
            entry["instruction"] = batch[i]["instruction"]
            entry["output"] = answer.strip() 
            entry["generator"] = generator_name
            entry["dataset"] = dataset_name
            results.append(entry)
        save_json(results, output_path) 
        print("saved", len(results), "of", len(prompts))
    print("done:", output_path)
