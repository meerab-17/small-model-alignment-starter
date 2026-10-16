import torch  
import random
from scripts.task3_sft import build_sft_batch, compute_response_logprobs, write_metric, save_checkpoint
import json
import os
import time

def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    batch = {}  
    batch["chosen"] = build_sft_batch(tokenizer, prompts, chosen_responses, max_length)
    batch["rejected"] = build_sft_batch(tokenizer, prompts, rejected_responses, max_length)
    return batch

def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,reference_chosen_logps, reference_rejected_logps, beta):
    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    x_values = beta * (policy_gap - reference_gap)
    pair_losses = []  
    for i in range(len(x_values)):
        x = x_values[i] 
        if x >= 0:
            pair_loss = torch.log(1 + torch.exp(-x))
        else:
            pair_loss = -x + torch.log(1 + torch.exp(x))
        pair_losses.append(pair_loss)
    return torch.stack(pair_losses).mean()

def get_dpo_logps(policy_model, reference_model, tokenizer, prompts,chosen_responses, rejected_responses, max_length):
    policy_chosen = compute_response_logprobs(policy_model, tokenizer, prompts, chosen_responses, max_length)
    policy_rejected = compute_response_logprobs(policy_model, tokenizer, prompts, rejected_responses, max_length)
    with torch.no_grad():
        reference_chosen = compute_response_logprobs(reference_model, tokenizer, prompts, chosen_responses, max_length)
        reference_rejected = compute_response_logprobs(reference_model, tokenizer, prompts, rejected_responses, max_length)
    device = policy_chosen.device
    reference_chosen = reference_chosen.to(device)
    reference_rejected = reference_rejected.to(device)
    return policy_chosen, policy_rejected, reference_chosen, reference_rejected

def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses,
                     rejected_responses, beta, max_length):
    pc, pr, rc, rr = get_dpo_logps(policy_model, reference_model, tokenizer, prompts,
                                   chosen_responses, rejected_responses, max_length)
    return compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)

def split_hh_transcript(text):
    human_marker = "\n\nHuman:"  
    assistant_marker = "\n\nAssistant:" 
    if text.count(human_marker) != 1 or text.count(assistant_marker) != 1:
        return None
    if not text.startswith(human_marker):  
        return None
    human_start = text.find(human_marker)  
    assistant_start = text.find(assistant_marker)  
    if assistant_start < human_start:  
        return None
    question = text[human_start + len(human_marker):assistant_start].strip()
    answer = text[assistant_start + len(assistant_marker):].strip()
    if question == "" or answer == "":  
        return None
    return question, answer

def extract_hh_pairs(rows, subset_name):
    from scripts.task2_baseline import make_prompt  
    pairs = []  
    skipped = 0  
    index = 0  
    for row in rows:
        chosen_parts = split_hh_transcript(row["chosen"])  
        rejected_parts = split_hh_transcript(row["rejected"]) 
        index = index + 1
        if chosen_parts is None or rejected_parts is None:  
            skipped = skipped + 1
            continue
        if chosen_parts[0] != rejected_parts[0]:  
            skipped = skipped + 1
            continue
        if chosen_parts[1] == rejected_parts[1]:  
            skipped = skipped + 1
            continue
        pair = {} 
        pair["id"] = subset_name + "_" + str(index - 1) 
        pair["prompt"] = make_prompt(chosen_parts[0])  
        pair["chosen"] = chosen_parts[1] 
        pair["rejected"] = rejected_parts[1] 
        pairs.append(pair)
    print(subset_name, "| usable pairs:", len(pairs), "| skipped:", skipped)
    return pairs


def extract_ultrafeedback_pairs(rows):
    from scripts.task2_baseline import make_prompt  
    pairs = []  
    skipped = 0  
    for row in rows:
        chosen = row["chosen"]  
        rejected = row["rejected"]  
        if len(chosen) != 2 or len(rejected) != 2:  
            skipped = skipped + 1
            continue
        if chosen[0]["role"] != "user" or chosen[1]["role"] != "assistant": 
            skipped = skipped + 1
            continue
        if rejected[0]["role"] != "user" or rejected[1]["role"] != "assistant":
            skipped = skipped + 1
            continue

        question = chosen[0]["content"].strip()  
        if question != rejected[0]["content"].strip():  
            skipped = skipped + 1
            continue
        chosen_answer = chosen[1]["content"].strip()  
        rejected_answer = rejected[1]["content"].strip()  
        if question == "" or chosen_answer == "" or rejected_answer == "": 
            skipped = skipped + 1
            continue
        if chosen_answer == rejected_answer:  
            skipped = skipped + 1
            continue

        pair = {}  
        pair["id"] = row["prompt_id"] 
        pair["prompt"] = make_prompt(question)  
        pair["chosen"] = chosen_answer
        pair["rejected"] = rejected_answer
        pairs.append(pair)

    print("ultrafeedback | usable pairs:", len(pairs), "| skipped:", skipped)
    return pairs


def pick_dpo_pairs(all_pairs, how_many, seed):
    rng = random.Random(seed)  
    order = list(range(len(all_pairs)))  
    rng.shuffle(order)  
    picked = []  
    for position in order:
        if len(picked) == how_many:  
            break
        picked.append(all_pairs[position])
    if len(picked) < how_many: 
        print("warning: asked for", how_many, "pairs but only", len(picked), "exist")
    print("picked:", len(picked))
    return picked


def drop_long_prompts(pairs, tokenizer, max_prompt_tokens):
    kept = []  
    dropped = 0  
    for pair in pairs:
        prompt_length = len(tokenizer.encode(pair["prompt"], add_special_tokens=False))
        if prompt_length > max_prompt_tokens:  
            dropped = dropped + 1
        else:
            kept.append(pair)  
    print("dropped for prompts over", max_prompt_tokens, "tokens:", dropped)
    return kept

def drop_overlapping_prompts(pool, train_pairs):
    train_prompts = {}  
    for pair in train_pairs:
        train_prompts[pair["prompt"]] = 1  
    kept = []  
    dropped = 0  
    for pair in pool:
        if pair["prompt"] in train_prompts:  
            dropped = dropped + 1
        else:
            kept.append(pair)
    print("dropped validation pairs whose prompt is in training:", dropped)
    return kept

def prepare_dpo_data(dataset_name, num_train, num_val, seed, tokenizer,max_prompt_tokens=256, hh_subsets=None):
    from datasets import load_dataset  
    if dataset_name == "hh":
        train_all = [] 
        val_all = []  
        for subset in hh_subsets:
            train_rows = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split="train")
            val_rows = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split="test") 
            train_all = train_all + extract_hh_pairs(train_rows, subset)
            val_all = val_all + extract_hh_pairs(val_rows, subset)
    elif dataset_name == "ultrafeedback":
        train_rows = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        val_rows = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")  
        train_all = extract_ultrafeedback_pairs(train_rows)
        val_all = extract_ultrafeedback_pairs(val_rows)
    else:
        print("unknown dataset name:", dataset_name)
        return None, None
    train_all = drop_long_prompts(train_all, tokenizer, max_prompt_tokens)
    val_all = drop_long_prompts(val_all, tokenizer, max_prompt_tokens)

    train_pairs = pick_dpo_pairs(train_all, num_train, seed)  
    val_all = drop_overlapping_prompts(val_all, train_pairs)  
    val_pairs = pick_dpo_pairs(val_all, num_val, seed)
    return train_pairs, val_pairs

def run_dpo_validation(policy_model, reference_model, tokenizer, val_pairs, beta, max_length, batch_size=4):
    policy_model.eval()  
    total_loss = 0.0  
    total_margin = 0.0  
    wins = 0  
    count = 0  
    with torch.no_grad():
        for start in range(0, len(val_pairs), batch_size):
            chunk = val_pairs[start:start + batch_size]
            prompts = []
            chosen = []
            rejected = []
            for pair in chunk:
                prompts.append(pair["prompt"])
                chosen.append(pair["chosen"])
                rejected.append(pair["rejected"])
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pc, pr, rc, rr = get_dpo_logps(policy_model, reference_model, tokenizer,prompts, chosen, rejected, max_length)

            batch_loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)  
            total_loss = total_loss + batch_loss.item() * len(chunk)  
            reward_chosen = beta * (pc - rc)  
            reward_rejected = beta * (pr - rr) 
            for i in range(len(chunk)):
                total_margin = total_margin + (reward_chosen[i] - reward_rejected[i]).item()
                if reward_chosen[i].item() > reward_rejected[i].item():
                    wins = wins + 1
            count = count + len(chunk)

    policy_model.train()  
    result = {}
    result["val_loss"] = total_loss / count
    result["val_reward_accuracy"] = wins / count  
    result["val_reward_margin"] = total_margin / count  
    return result

def train_dpo(policy_model, reference_model, tokenizer, train_pairs, val_pairs, output_dir,
              beta=0.1, lr=1e-6, weight_decay=0.0, clip_norm=1.0,
              micro_batch_size=1, accum_steps=32, max_length=512,
              eval_every=50, val_batch_size=4):
    os.makedirs(output_dir, exist_ok=True)  
    total_steps = (len(train_pairs) // micro_batch_size) // accum_steps
    print("optimizer steps:", total_steps)
    for parameter in reference_model.parameters():
        parameter.requires_grad = False
    reference_model.eval()  
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda")  

    config = {  
        "beta": beta, "lr": lr, "lr_schedule": "fixed", "weight_decay": weight_decay,
        "clip_norm": clip_norm, "micro_batch_size": micro_batch_size, "accum_steps": accum_steps,
        "effective_batch_size": micro_batch_size * accum_steps, "max_length": max_length,
        "total_steps": total_steps, "train_pairs": len(train_pairs), "val_pairs": len(val_pairs),
        "eval_every": eval_every, "optimizer": "AdamW",
                "precision": "policy fp32 weights with fp16 autocast, reference " + str(next(reference_model.parameters()).dtype),
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    open(metrics_path, "w").close()  

    policy_model.config.use_cache = False  
    policy_model.train()  # training mode
    result = run_dpo_validation(policy_model, reference_model, tokenizer, val_pairs,
                                beta, max_length, val_batch_size)
    print("step 0 | validation:", result)
    record = {"type": "val", "step": 0}
    record.update(result)  
    write_metric(metrics_path, record)

    best_accuracy = None  
    micro_index = 0  
    start_time = time.time()

    for step in range(total_steps):
        optimizer.zero_grad(set_to_none=True)  
        group_loss = 0.0  
        group_wins = 0  
        group_margin = 0.0  
        pairs_seen = 0  
        for k in range(accum_steps):
            begin = micro_index * micro_batch_size
            chunk = train_pairs[begin:begin + micro_batch_size]  
            micro_index = micro_index + 1
            prompts = []
            chosen = []
            rejected = []
            for pair in chunk:
                prompts.append(pair["prompt"])
                chosen.append(pair["chosen"])
                rejected.append(pair["rejected"])

            with torch.autocast(device_type="cuda", dtype=torch.float16):  
                pc, pr, rc, rr = get_dpo_logps(policy_model, reference_model, tokenizer,
                                               prompts, chosen, rejected, max_length)
            loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)  

            if loss.requires_grad:  
                scaler.scale(loss / accum_steps).backward()  
            group_loss = group_loss + loss.item() / accum_steps

            reward_chosen = (beta * (pc - rc)).detach()  
            reward_rejected = (beta * (pr - rr)).detach()
            for i in range(len(chunk)):
                group_margin = group_margin + (reward_chosen[i] - reward_rejected[i]).item()
                if reward_chosen[i].item() > reward_rejected[i].item():
                    group_wins = group_wins + 1
                pairs_seen = pairs_seen + 1

        scaler.unscale_(optimizer)  
        grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), clip_norm)  
        scaler.step(optimizer)  
        scaler.update()  

        step_number = step + 1  
        write_metric(metrics_path, {"type": "train", "step": step_number, "loss": group_loss,
                                    "reward_accuracy": group_wins / pairs_seen,
                                    "reward_margin": group_margin / pairs_seen,
                                    "grad_norm": float(grad_norm)})
        if step_number % 10 == 0 or step_number == 1:
            minutes = (time.time() - start_time) / 60
            print("step", step_number, "of", total_steps, "| loss", round(group_loss, 4),
                  "| reward acc", round(group_wins / pairs_seen, 3),
                  "| margin", round(group_margin / pairs_seen, 4), "| minutes", round(minutes, 1))
        if step_number == 1:
            print("peak memory on gpu 0 (GB):", torch.cuda.max_memory_allocated(0) / 1e9)
        if step_number % eval_every == 0 or step_number == total_steps:
            result = run_dpo_validation(policy_model, reference_model, tokenizer, val_pairs,
                                        beta, max_length, val_batch_size)
            print("step", step_number, "| validation:", result)
            record = {"type": "val", "step": step_number}
            record.update(result)
            write_metric(metrics_path, record)
            if best_accuracy is None or result["val_reward_accuracy"] > best_accuracy:
                best_accuracy = result["val_reward_accuracy"]
                save_checkpoint(policy_model, tokenizer, os.path.join(output_dir, "best"))
                print("saved new best checkpoint")

    save_checkpoint(policy_model, tokenizer, os.path.join(output_dir, "final"))  
    print("done. best validation reward accuracy:", best_accuracy)
    return best_accuracy
