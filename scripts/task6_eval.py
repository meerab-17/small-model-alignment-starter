import gc  
import json  
import os 
import random  
import torch  


def generate_with_checkpoint(checkpoint_path, generator_name, jobs, limit=None):
    from scripts import task2_baseline  
    model, tokenizer = task2_baseline.load_model_and_tokenizer(checkpoint_path)
    for job in jobs:
        prompts_path = job[0] 
        dataset_name = job[1]  
        output_path = "outputs/generations/" + generator_name + "_" + dataset_name + ".json"
        task2_baseline.generate_answers(model, tokenizer, checkpoint_path, prompts_path, output_path,
                                        generator_name, dataset_name=dataset_name, limit=limit)

    del model  
    gc.collect()
    torch.cuda.empty_cache()

def read_answers(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) 


def build_pairs(file_a, name_a, file_b, name_b, output_path, seed):
    answers_a = read_answers(file_a)  # answers of the first model
    answers_b = read_answers(file_b)  # answers of the second model, to the same prompts

    lookup_b = {}  # find the second model's answer by prompt id
    for item in answers_b:
        lookup_b[item["id"]] = item

    rng = random.Random(seed)  
    lines = []  
    count_a_first = 0  
    missing = 0  # prompts that only one of the two models answered

    for item in answers_a:
        if item["id"] not in lookup_b:
            missing = missing + 1
            continue
        other = lookup_b[item["id"]]

        pair = {}  # one comparison
        pair["id"] = item["id"]
        pair["instruction"] = item["instruction"]
        if rng.random() < 0.5:  # a coin flip decides which model is shown as response A
            pair["response_a"] = item["output"]
            pair["response_b"] = other["output"]
            pair["hidden_mapping"] = {"A": name_a, "B": name_b}  # kept for later, never shown to the judge
            count_a_first = count_a_first + 1
        else:
            pair["response_a"] = other["output"]
            pair["response_b"] = item["output"]
            pair["hidden_mapping"] = {"A": name_b, "B": name_a}
        lines.append(pair)

    folder = os.path.dirname(output_path)
    os.makedirs(folder, exist_ok=True)  # make the folder if it is missing
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in lines:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")  # one pair per line

    print("pairs saved:", len(lines), "| missing in one file:", missing, "|", name_a, "shown as A:", count_a_first)
    return lines


def show_answers(path, how_many):
    answers = read_answers(path)
    for item in answers[:how_many]:
        print("PROMPT:", item["instruction"][:200])
        print("ANSWER:", item["output"][:400])
        print("-----")
def answer_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        answers = json.load(f)  # a list of {id, instruction, output, ...}

    empty = 0  
    very_long = 0  # answers over 900 characters, a rough sign they ran to the length cap
    looping = 0  
    total_chars = 0  # used for the average length

    for item in answers:
        text = item["output"]
        total_chars = total_chars + len(text)
        if text == "":
            empty = empty + 1
        if len(text) > 900:
            very_long = very_long + 1

        seen = {}  
        is_looping = False
        for line in text.split("\n"):
            cleaned = line.strip().lstrip("0123456789. ")  
            if cleaned == "":
                continue
            if cleaned in seen:
                seen[cleaned] = seen[cleaned] + 1
            else:
                seen[cleaned] = 1
            if seen[cleaned] >= 3:
                is_looping = True
        if is_looping:
            looping = looping + 1

    stats = {}
    stats["answers"] = len(answers)
    stats["empty"] = empty
    stats["over_900_chars"] = very_long
    stats["line_repeated_3_times"] = looping
    if len(answers) > 0:
        stats["average_chars"] = round(total_chars / len(answers), 1)
    else:
        stats["average_chars"] = 0
    return stats


def summarize_generations(folder, output_path):
    summary = {}  # file name -> stats
    for name in sorted(os.listdir(folder)):
        if name.endswith(".json") and not name.endswith(".config.json"):  
            summary[name] = answer_stats(os.path.join(folder, name))
            print(name, summary[name])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
