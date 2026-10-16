import json  # for reading the metrics lines
import os  # for making the output folder
import matplotlib.pyplot as plt  # for drawing the plots


def read_metrics(path):
    records = []  # one dictionary per line of the metrics file
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))  # turn the text of the line into a dictionary
    return records


def moving_average(values, window):
    result = []  # the smoothed values
    for i in range(len(values)):
        start = i - window + 1  # the window covers the last `window` values
        if start < 0:
            start = 0  # near the beginning there are fewer values to average
        total = 0.0
        for j in range(start, i + 1):
            total = total + values[j]
        result.append(total / (i + 1 - start))
    return result


def plot_sft_curves(metrics_path, save_path):
    records = read_metrics(metrics_path)

    train_steps = []  # step numbers of the training records
    train_loss = []  # training loss per update
    learning_rates = []  # learning rate per update
    grad_norms = []  # gradient norm per update
    val_steps = []  # step numbers of the validation records
    val_loss = []  # validation loss
    for record in records:
        if record["type"] == "train":
            train_steps.append(record["step"])
            train_loss.append(record["loss"])
            learning_rates.append(record["lr"])
            grad_norms.append(record["grad_norm"])
        else:
            val_steps.append(record["step"])
            val_loss.append(record["val_loss"])

    best_step = val_steps[val_loss.index(min(val_loss))]  # the validation point with the lowest loss

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # three plots side by side

    # plot 1: training and validation loss
    axes[0].plot(train_steps, train_loss, color="lightgray", label="train loss (raw)")
    axes[0].plot(train_steps, moving_average(train_loss, 20), color="tab:blue", label="train loss (20-step average)")
    axes[0].plot(val_steps, val_loss, color="tab:red", marker="o", label="validation loss")
    axes[0].axvline(best_step, color="green", linestyle="--", label="best checkpoint (step " + str(best_step) + ")")
    axes[0].set_ylim(min(train_loss) - 0.05, max(train_loss) * 1.3)  # empty room at the top for the legend
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("loss per response token")
    axes[0].set_title("SFT loss")
    axes[0].legend(loc="upper right", fontsize=8)

    # plot 2: learning rate
    axes[1].plot(train_steps, learning_rates, color="tab:orange")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("warmup then cosine decay")

    # plot 3: gradient norm before clipping, with the clip limit as a line
    axes[2].plot(train_steps, grad_norms, color="tab:purple", label="gradient norm")
    axes[2].axhline(1.0, color="black", linestyle="--", label="clip limit (1.0)")
    axes[2].set_ylim(0, max(grad_norms) * 1.35)  # empty room at the top for the legend
    axes[2].set_xlabel("optimizer step")
    axes[2].set_ylabel("gradient norm")
    axes[2].set_title("gradient norm")
    axes[2].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # make the folder if it is missing
    plt.savefig(save_path, dpi=150)  # save the picture into the run's folder
    plt.show()  # also show it in the notebook log
    plt.close(fig)  # free the memory
    print("saved plot to", save_path)


def plot_dpo_curves(metrics_path, save_path):
    records = read_metrics(metrics_path)

    train_steps = []  # step numbers of the training records
    train_loss = []  # training loss per update
    train_acc = []  # training reward accuracy per update
    train_margin = []  # training reward margin per update
    grad_norms = []  # gradient norm per update
    val_steps = []  # step numbers of the validation records
    val_loss = []  # validation loss
    val_acc = []  # validation reward accuracy
    val_margin = []  # validation reward margin
    for record in records:
        if record["type"] == "train":
            train_steps.append(record["step"])
            train_loss.append(record["loss"])
            train_acc.append(record["reward_accuracy"])
            train_margin.append(record["reward_margin"])
            grad_norms.append(record["grad_norm"])
        else:
            val_steps.append(record["step"])
            val_loss.append(record["val_loss"])
            val_acc.append(record["val_reward_accuracy"])
            val_margin.append(record["val_reward_margin"])

    best_step = val_steps[val_acc.index(max(val_acc))]  # the checkpoint with the highest validation accuracy

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))  # a 2 by 2 grid of plots

    # loss
    axes[0][0].plot(train_steps, moving_average(train_loss, 20), color="tab:blue", label="train (20-step average)")
    axes[0][0].plot(val_steps, val_loss, color="tab:red", marker="o", label="validation")
    axes[0][0].axvline(best_step, color="green", linestyle="--", label="best checkpoint (step " + str(best_step) + ")")
    axes[0][0].set_xlabel("optimizer step")
    axes[0][0].set_ylabel("DPO loss")
    axes[0][0].set_title("DPO loss")
    axes[0][0].legend(fontsize=8)

    # reward accuracy
    axes[0][1].plot(train_steps, moving_average(train_acc, 20), color="tab:blue", label="train (20-step average)")
    axes[0][1].plot(val_steps, val_acc, color="tab:red", marker="o", label="validation")
    axes[0][1].axvline(best_step, color="green", linestyle="--")
    axes[0][1].set_xlabel("optimizer step")
    axes[0][1].set_ylabel("reward accuracy")
    axes[0][1].set_title("reward accuracy")
    axes[0][1].legend(fontsize=8)

    # reward margin
    axes[1][0].plot(train_steps, moving_average(train_margin, 20), color="tab:blue", label="train (20-step average)")
    axes[1][0].plot(val_steps, val_margin, color="tab:red", marker="o", label="validation")
    axes[1][0].axvline(best_step, color="green", linestyle="--")
    axes[1][0].set_xlabel("optimizer step")
    axes[1][0].set_ylabel("reward margin")
    axes[1][0].set_title("reward margin")
    axes[1][0].legend(fontsize=8)

    # gradient norm before clipping, with the clip limit as a line
    axes[1][1].plot(train_steps, grad_norms, color="tab:purple", label="gradient norm")
    axes[1][1].axhline(1.0, color="black", linestyle="--", label="clip limit (1.0)")
    axes[1][1].set_ylim(0, max(grad_norms) * 1.35)  # empty room at the top for the legend
    axes[1][1].set_xlabel("optimizer step")
    axes[1][1].set_ylabel("gradient norm")
    axes[1][1].set_title("gradient norm")
    axes[1][1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # make the folder if it is missing
    plt.savefig(save_path, dpi=150)  # save the picture into the run's folder
    plt.show()  # also show it in the notebook log
    plt.close(fig)  # free the memory
    print("saved plot to", save_path)
def plot_lora_curves(metrics_path, save_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        log = json.load(f)  # the trainer's log, a list of dictionaries

    train_steps = []  # step numbers of the training logs
    train_loss = []  # training loss
    train_acc = []  # training reward accuracy
    train_margin = []  # training reward margin
    val_steps = []  # step numbers of the validation logs
    val_loss = []  # validation loss
    val_acc = []  # validation reward accuracy
    val_margin = []  # validation reward margin

    counter = 0  # used when a log line has no step number
    for entry in log:
        counter = counter + 1
        step = entry.get("step", counter)  # the step number if the log has one
        if "eval_loss" in entry:  # a validation log
            val_steps.append(step)
            val_loss.append(entry["eval_loss"])
            val_acc.append(entry["eval_rewards/accuracies"])
            val_margin.append(entry["eval_rewards/margins"])
        elif "loss" in entry and "rewards/accuracies" in entry:  # a training log
            train_steps.append(step)
            train_loss.append(entry["loss"])
            train_acc.append(entry["rewards/accuracies"])
            train_margin.append(entry["rewards/margins"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # three plots side by side

    axes[0].plot(train_steps, train_loss, color="tab:blue", label="train")
    axes[0].plot(val_steps, val_loss, color="tab:red", marker="o", label="validation (20 pairs)")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("DPO loss")
    axes[0].set_title("LoRA DPO loss")
    axes[0].legend(fontsize=8)

    axes[1].plot(train_steps, train_acc, color="tab:blue", label="train")
    axes[1].plot(val_steps, val_acc, color="tab:red", marker="o", label="validation (20 pairs)")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("reward accuracy")
    axes[1].set_title("reward accuracy")
    axes[1].legend(fontsize=8, loc="lower right")

    axes[2].plot(train_steps, train_margin, color="tab:blue", label="train")
    axes[2].plot(val_steps, val_margin, color="tab:red", marker="o", label="validation (20 pairs)")
    axes[2].set_xlabel("optimizer step")
    axes[2].set_ylabel("reward margin")
    axes[2].set_title("reward margin")
    axes[2].legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # make the folder if it is missing
    plt.savefig(save_path, dpi=150)  # save the picture into the run's folder
    plt.show()  # also show it in the notebook log
    plt.close(fig)  # free the memory
    print("saved plot to", save_path)
