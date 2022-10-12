import argparse
from functools import partial
import os
import random

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import default_data_collator, get_scheduler, AutoConfig, AutoTokenizer
import datasets
from datasets import load_dataset
from nltk import wordpunct_tokenize

import numpy as np
from sklearn.metrics import f1_score
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from models import BERT_CON, BERT_SCL, BERT_STL

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(args, tokenizer, batch_size):
    data_files = {}
    data_files["train"] = os.path.join(args.data_dir, "train.csv")
    data_files["validation"] = os.path.join(args.data_dir, "dev.csv")
    data_files["test"] = os.path.join(args.data_dir, "test.csv")

    raw_datasets = load_dataset("csv", data_files=data_files)

    label_list = raw_datasets["train"].unique("label")
    label_list.sort()
    label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function_2(examples):
        # Tokenize the texts
        texts = (examples["text"],)
        result = tokenizer(
            *texts, padding="max_length", max_length=args.max_length, truncation=True
        )

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]

        return result

    def preprocess_function(examples):
        input_ids_list = []
        target_mask_list = []
        target_idx_list = []
        attention_mask_list = []

        input_ids_2_list = []
        target_mask_2_list = []
        attention_mask_2_list = []

        labels = []

        for idx in range(len(examples["text"])):
            label = examples["label"][idx]
            if label_to_id:
                labels.append(label_to_id[label])
            else:
                labels.append(label)

            tokens_a = tokenizer.tokenize(examples["text"][idx])
            tokens_b = None

            text_b = int(examples["target_index"][idx])
            tokens_b = text_b

            # truncate the sentence to max_seq_len
            if len(tokens_a) > args.max_length - 2:
                tokens_a = tokens_a[: (args.max_length - 2)]

            # Find the target word index
            for i, w in enumerate(wordpunct_tokenize(examples["text"][idx])):
                # If w is a target word, tokenize the word and save to text_b
                if i == text_b:
                    # consider the index due to models that use a byte-level BPE as a tokenizer (e.g., GPT2, RoBERTa)
                    text_b = (
                        tokenizer.tokenize(w) if i == 0 else tokenizer.tokenize(" " + w)
                    )
                    break
                w_tok = tokenizer.tokenize(w) if i == 0 else tokenizer.tokenize(" " + w)

                # Count number of tokens before the target word to get the target word index
                if w_tok:
                    tokens_b += len(w_tok) - 1

            tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token]
            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            try:
                tokens_b += 1  # add 1 to the target word index considering [CLS]
                for i in range(len(text_b)):
                    segment_ids[tokens_b + i] = 1
            except:
                print(examples["post_id"][i])

            target_idx_list.append(tokens_b)

            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)] * (
                args.max_length - len(input_ids)
            )
            input_ids += padding
            input_mask += [0] * len(padding)
            segment_ids += [0] * len(padding)

            assert len(input_ids) == args.max_length
            assert len(input_mask) == args.max_length
            assert len(segment_ids) == args.max_length

            input_ids_list.append(input_ids)
            attention_mask_list.append(input_mask)
            target_mask_list.append(segment_ids)

            # Second features (Target word)
            tokens = [tokenizer.cls_token] + text_b + [tokenizer.sep_token]
            segment_ids_2 = [0] * len(tokens)
            try:
                tokens_b = 1  # add 1 to the target word index considering [CLS]
                for i in range(len(text_b)):
                    segment_ids_2[tokens_b + i] = 1
            except TypeError:
                pass

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_ids_2 = tokenizer.convert_tokens_to_ids(tokens)
            input_mask_2 = [1] * len(input_ids_2)

            padding = [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)] * (
                args.max_length - len(input_ids_2)
            )
            input_ids_2 += padding
            input_mask_2 += [0] * len(padding)
            segment_ids_2 += [0] * len(padding)

            assert len(input_ids_2) == args.max_length
            assert len(input_mask_2) == args.max_length
            assert len(segment_ids_2) == args.max_length

            input_ids_2_list.append(input_ids_2)
            target_mask_2_list.append(segment_ids_2)
            attention_mask_2_list.append(input_mask_2)

        result = {
            "input_ids": input_ids_list,
            "input_ids_2": input_ids_2_list,
            #             "target_mask": target_mask_list,
            #             "target_mask_2": target_mask_2_list,
            "attention_mask_2": attention_mask_2_list,
            #             "token_type_ids": target_mask_list,
            "target_index": target_idx_list,
            "labels": labels,
        }
        return result

    if args.model == "bert":
        processed_datasets = raw_datasets.map(
                    preprocess_function_2,
                    batched=True,
                    remove_columns=raw_datasets["train"].column_names,
                    desc="Running tokenizer on dataset",
                )
    else:
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    data_collator = default_data_collator

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    return train_dataloader, eval_dataloader, label_list


def build_optimizer(model, learning_rate, max_train_steps):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )

    return optimizer, lr_scheduler


def model_init(args, label_list, alpha=0.2, temperature=0.3):
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
    )
    if args.model == "con":
        model = BERT_CON(
            args.model_name_or_path,
            len(label_list),
            alpha=alpha,
        )
    elif args.model == "scl":
        model = BERT_SCL(
                args.model_name_or_path,
                len(label_list),
                alpha=alpha,
                temperature=temperature,
            )
    elif args.model == "bert":
        model = BERT_STL(
                args.model_name_or_path,
                len(label_list),
                # args.dropout,
            )

    model.enc_model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.enc_model.config.id2label = {
        id: label for label, id in config.label2id.items()
    }

    return model

def train(config, args):
    epochs = 5

    # For seed results
    results = {}
    seeds = [42, 52, 62, 72, 100]

    for seed in seeds:
        print(f"Running training with seed = {seed}")
        set_seed(seed)

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        train_dataloader, eval_dataloader, label_list = build_dataset(
            args, tokenizer, config["batch_size"]
        )
        
        if args.model == "con":
            model = model_init(args, label_list, alpha=config["alpha"])
            model.to(device)
        elif args.model == "scl":
            model = model_init(args, label_list, alpha=config["alpha"], temperature=config["temperature"])
            model.to(device)
        elif args.model == "bert":
            model = model_init(args, label_list)
            model.to(device)

        num_update_steps_per_epoch = len(train_dataloader)
        max_train_steps = epochs * num_update_steps_per_epoch
        optimizer, lr_scheduler = build_optimizer(
            model, config["learning_rate"], max_train_steps
        )

        best_eval_f1 = 0
        for epoch in range(epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch = {key: batch[key].to(device) for key in batch}
                outputs = model(**batch)
                loss = outputs[0]
                loss.backward()
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            model.eval()
            eval_loss = 0.0
            y_pred = None
            y_true = None
            for step, batch in enumerate(eval_dataloader):
                batch = {key: batch[key].to(device) for key in batch}
                with torch.no_grad():
                    outputs = model(**batch)

                eval_loss += outputs[0].item()

                if y_pred is None:
                    y_pred = outputs[1].argmax(dim=-1).detach().cpu().numpy()
                    y_true = batch["labels"].detach().cpu().numpy()
                else:
                    y_pred = np.append(
                        y_pred, outputs[1].argmax(dim=-1).detach().cpu().numpy(), axis=0
                    )
                    y_true = np.append(y_true, batch["labels"].detach().cpu().numpy())

            eval_loss = eval_loss / len(eval_dataloader)
            eval_f1 = f1_score(y_true, y_pred, average="macro")

            if eval_f1 > best_eval_f1:
                best_eval_f1 = eval_f1

        print(f"Seed = {seed}, Eval F1 = {best_eval_f1}")
        results[seed] = best_eval_f1

    avg_f1 = sum(results[k] for k in results) / len(results)
    print(f"Average F1 {avg_f1}")
    tune.report(eval_f1=avg_f1)

    print("Finished Training")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune model on HMC task"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing training and validation data",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["bert", "con", "scl"],
        required=True,
        help="the name of the model to use. some models may use different model args than others.",
    )
    parser.add_argument(
        "--num_cpu", type=int, default=1, help="Number of cpu"
    )
    parser.add_argument(
        "--num_trials", type=int, default=20, help="Number of trials"
    )
    parser.add_argument(
        "--max_num_epochs", type=int, default=5, help="Max no of epochs"
    )
    parser.add_argument(
        "--gpus_per_trial", type=int, default=1, help="Max no of GPUs per trial"
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    args.data_dir = os.path.abspath(args.data_dir)

    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    # hyperparameters
    if args.model == "con":
        config = {
            "batch_size": tune.choice([8, 16, 32]),
            "learning_rate": tune.choice([1e-5, 2e-5, 3e-5]),
            "alpha": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        }
    elif args.model == "scl":
        config = {
            "batch_size": tune.choice([8, 16, 32]),
            "learning_rate": tune.choice([1e-5, 2e-5, 3e-5]),
            "alpha": tune.choice([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
            "temperature": tune.choice([0.1, 0.3, 0.5, 0.7]),
        }
    elif args.model == "bert":
        config = {
            "batch_size": tune.choice([8, 16, 32]),
            "learning_rate": tune.choice([1e-5, 2e-5, 3e-5]),
        }


    scheduler = ASHAScheduler(
        metric="eval_f1",
        mode="max",
        max_t=args.max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    reporter = CLIReporter(metric_columns=["eval_f1", "training_iteration"])
    result = tune.run(
        partial(train, args=args),
        resources_per_trial={"cpu": args.num_cpu, "gpu": args.gpus_per_trial},
        config=config,
        num_samples=args.num_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    best_trial = result.get_best_trial("eval_f1", "max", "last")

    print("Best trial config: {}".format(best_trial.config))
    print(
        "Best trial final validation F1: {}".format(best_trial.last_result["eval_f1"])
    )


if __name__ == "__main__":
    main()
