import torch
import random 
import math
import json
import time
import os

def build_sft_batch(tokenizer, prompts, responses, max_length):
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id
    if pad_id is None:
        pad_id = 0

    all_ids = []  
    all_labels = []  
    for i in range(len(prompts)):
        prompt_ids = tokenizer.encode(prompts[i], add_special_tokens=False)
        response_ids = tokenizer.encode(responses[i], add_special_tokens=False)
        if eos_id is not None:
            response_ids = response_ids + [eos_id]
        ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        ids = ids[:max_length]
        labels = labels[:max_length]
        all_ids.append(ids)
        all_labels.append(labels)
    longest = 0
    for ids in all_ids:
        if len(ids) > longest:
            longest = len(ids)

    input_ids = []
    attention_mask = [] 
    labels_padded = []  
    for i in range(len(all_ids)):
        ids = all_ids[i]
        labels = all_labels[i]
        pad_count = longest - len(ids)  
        input_ids.append(ids + [pad_id] * pad_count)
        attention_mask.append([1] * len(ids) + [0] * pad_count)
        labels_padded.append(labels + [-100] * pad_count)
    batch = {}
    batch["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
    batch["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
    batch["labels"] = torch.tensor(labels_padded, dtype=torch.long)
    return batch

def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels_list = batch["labels"].tolist()
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    row_logps = []  
    for row in range(len(labels_list)):
        labels = labels_list[row]  
        positions = []  
        targets = []  
        for t in range(len(labels) - 1):
            next_label = labels[t + 1]  
            if next_label != -100:  
                positions.append(t)
                targets.append(next_label)
        if len(positions) == 0:
            row_logps.append(torch.zeros((), device=device))
            continue
        position_tensor = torch.tensor(positions, device=device)  # positions as a tensor
        target_tensor = torch.tensor(targets, device=device)  # true tokens as a tensor
        row_logits = logits[row][position_tensor]
        row_logits = row_logits.float()  
        max_scores = row_logits.max(dim=-1, keepdim=True).values
        shifted = row_logits - max_scores
        sum_exp = torch.exp(shifted).sum(dim=-1)
        log_denominator = torch.log(sum_exp)
        counter = torch.arange(len(positions), device=device)
        true_scores = shifted[counter, target_tensor]
        token_logps = true_scores - log_denominator
        row_logps.append(token_logps.sum())
    return torch.stack(row_logps)

def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    response_logps = compute_response_logprobs(model, tokenizer, prompts, responses, max_length)
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    labels_list = batch["labels"].tolist() 

    token_count = 0  
    for labels in labels_list:
        for t in range(1, len(labels)):  
            if labels[t] != -100:  
                token_count = token_count + 1
    if token_count == 0:
        return response_logps.sum() * 0.0
    loss = -response_logps.sum() / token_count
    return loss
def make_pair(row):
    from scripts.task2_baseline import make_prompt

    messages = row["messages"] 
    if len(messages) < 2:  
        return None
    first = messages[0]  
    second = messages[1]  
    if first["role"] != "user" or second["role"] != "assistant":  
        return None
    question = first["content"].strip() 
    answer = second["content"].strip()
    if question == "" or answer == "":  
        return None

    pair = {}  
    pair["id"] = row["prompt_id"]  
    pair["prompt"] = make_prompt(question)  
    pair["response"] = answer  
    return pair


def pick_pairs(data, how_many, seed):
    rng = random.Random(seed)  
    positions = list(range(len(data))) 
    rng.shuffle(positions)  

    pairs = []  
    skipped = 0  
    for pos in positions:
        if len(pairs) == how_many: 
            break
        pair = make_pair(data[pos])  
        if pair is None:
            skipped = skipped + 1  
        else:
            pairs.append(pair)  

    print("kept:", len(pairs), "| skipped:", skipped)  
    return pairs


def prepare_sft_data(num_train, num_val, seed):
    from datasets import load_dataset  

    train_data = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    val_data = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    train_pairs = pick_pairs(train_data, num_train, seed)
    val_pairs = pick_pairs(val_data, num_val, seed)
    return train_pairs, val_pairs

def get_learning_rate(step, total_steps, peak_lr, warmup_fraction):
    warmup_steps = int(total_steps * warmup_fraction)
    if warmup_steps < 1:
        warmup_steps = 1
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / decay_steps
    if progress > 1.0:  
        progress = 1.0
    return peak_lr * 0.5 * (1 + math.cos(math.pi * progress))

def count_response_tokens(tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    labels_list = batch["labels"].tolist()  
    count = 0
    for labels in labels_list:
        for t in range(1, len(labels)):  
            if labels[t] != -100:  
                count = count + 1
    return count


def run_validation(model, tokenizer, val_pairs, batch_size, max_length):
    model.eval() 
    total_nll = 0.0  
    total_tokens = 0  
    with torch.no_grad():  
        for start in range(0, len(val_pairs), batch_size):
            chunk = val_pairs[start:start + batch_size]  
            prompts = []
            responses = []
            for pair in chunk:
                prompts.append(pair["prompt"])
                responses.append(pair["response"])
            with torch.autocast(device_type="cuda", dtype=torch.float16):  
                logps = compute_response_logprobs(model, tokenizer, prompts, responses, max_length)
            total_nll = total_nll - logps.sum().item() 
            total_tokens = total_tokens + count_response_tokens(tokenizer, prompts, responses, max_length)
    model.train()  
    if total_tokens == 0:  
        return 0.0
    return total_nll / total_tokens  

def write_metric(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def save_checkpoint(model, tokenizer, folder):
    os.makedirs(folder, exist_ok=True)  
    model.save_pretrained(folder)  
    tokenizer.save_pretrained(folder)  

def train_sft(model, tokenizer, train_pairs, val_pairs, output_dir,
              peak_lr=2e-5, warmup_fraction=0.03, weight_decay=0.1, clip_norm=1.0,
              micro_batch_size=2, accum_steps=16, max_length=512,
              eval_every=100, val_batch_size=4):
    os.makedirs(output_dir, exist_ok=True)  
    micro_batches = len(train_pairs) // micro_batch_size
    total_steps = micro_batches // accum_steps
    print("optimizer steps:", total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda")  
    config = {  
        "peak_lr": peak_lr, "warmup_fraction": warmup_fraction, "weight_decay": weight_decay,
        "clip_norm": clip_norm, "micro_batch_size": micro_batch_size, "accum_steps": accum_steps,
        "effective_batch_size": micro_batch_size * accum_steps, "max_length": max_length,
        "total_steps": total_steps, "train_pairs": len(train_pairs), "val_pairs": len(val_pairs),
        "eval_every": eval_every, "optimizer": "AdamW", "precision": "fp16 autocast, fp32 weights",
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    open(metrics_path, "w").close()  

    model.config.use_cache = False  
    model.train()  
    val_loss = run_validation(model, tokenizer, val_pairs, val_batch_size, max_length)
    print("step 0 | validation loss:", val_loss)
    write_metric(metrics_path, {"type": "val", "step": 0, "val_loss": val_loss})

    best_val = None  
    micro_index = 0  
    start_time = time.time()
    for step in range(total_steps):
        lr = get_learning_rate(step, total_steps, peak_lr, warmup_fraction)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)  
        group_loss = 0.0  
        for k in range(accum_steps):
            begin = micro_index * micro_batch_size
            chunk = train_pairs[begin:begin + micro_batch_size]  
            micro_index = micro_index + 1
            prompts = []
            responses = []
            for pair in chunk:
                prompts.append(pair["prompt"])
                responses.append(pair["response"])
            with torch.autocast(device_type="cuda", dtype=torch.float16):  
                loss = compute_sft_loss(model, tokenizer, prompts, responses, max_length)
            if loss.requires_grad:  
                scaler.scale(loss / accum_steps).backward()  
            group_loss = group_loss + loss.item() / accum_steps
        scaler.unscale_(optimizer)  # undo the scaling so clipping sees the true gradient size
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)  # shrink huge gradients
        scaler.step(optimizer)  
        scaler.update()  
        step_number = step + 1  
        write_metric(metrics_path, {"type": "train", "step": step_number, "loss": group_loss,
                                    "lr": lr, "grad_norm": float(grad_norm)})
        if step_number % 10 == 0 or step_number == 1:
            minutes = (time.time() - start_time) / 60
            print("step", step_number, "of", total_steps, "| loss", round(group_loss, 4),
                  "| lr", lr, "| minutes", round(minutes, 1))
        if step_number == 1:
            print("peak gpu memory so far (GB):", torch.cuda.max_memory_allocated() / 1e9)

        if step_number % eval_every == 0 or step_number == total_steps:
            val_loss = run_validation(model, tokenizer, val_pairs, val_batch_size, max_length)
            print("step", step_number, "| validation loss:", val_loss)
            write_metric(metrics_path, {"type": "val", "step": step_number, "val_loss": val_loss})
            if best_val is None or val_loss < best_val:  
                best_val = val_loss
                save_checkpoint(model, tokenizer, os.path.join(output_dir, "best"))
                print("saved new best checkpoint")

    save_checkpoint(model, tokenizer, os.path.join(output_dir, "final")) 
    print("done. best validation loss:", best_val)
    return best_val
