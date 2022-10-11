import logging
import os

import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import transformers
from transformers import default_data_collator, get_scheduler, AutoConfig, AutoTokenizer
import evaluate
from tqdm.auto import tqdm

import datasets
from datasets import load_dataset
from nltk import wordpunct_tokenize

import torch
from torch.utils.data import DataLoader

from models import BERT_CON

logger = get_logger(__name__)

os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_PROJECT"] = "nairaland-hmc"
os.environ["WANDB_LOG_MODEL"] = "true"

metric = evaluate.load("f1")

def build_dataset(data_path, tokenizer, batch_size):
    data_files = {}
    data_files["train"] = data_path + "train.csv"
    data_files["validation"] = data_path + "dev.csv"
    data_files["test"] = data_path + "test.csv"

    raw_datasets = load_dataset("csv", data_files=data_files)

    label_list = raw_datasets["train"].unique("label")
    label_list.sort()
    label_to_id = {v: i for i, v in enumerate(label_list)}

    padding = "max_length"
    max_seq_length = 128

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
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[: (max_seq_length - 2)]

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
                max_seq_length - len(input_ids)
            )
            input_ids += padding
            input_mask += [0] * len(padding)
            segment_ids += [0] * len(padding)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

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
                max_seq_length - len(input_ids_2)
            )
            input_ids_2 += padding
            input_mask_2 += [0] * len(padding)
            segment_ids_2 += [0] * len(padding)

            assert len(input_ids_2) == max_seq_length
            assert len(input_mask_2) == max_seq_length
            assert len(segment_ids_2) == max_seq_length

            input_ids_2_list.append(input_ids_2)
            target_mask_2_list.append(segment_ids_2)
            attention_mask_2_list.append(input_mask_2)

        #         import pdb; pdb.set_trace()

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

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    #     test_dataset = processed_datasets["test"]

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


def model_init(model_name_or_path, label_list, lam):
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(label_list),
    )

    model = BERT_CON(
        model_name_or_path,
        len(label_list),
        lam=lam,
    )
    model.enc_model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.enc_model.config.id2label = {
        id: label for label, id in config.label2id.items()
    }

    return model


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # For seed results
        results = {}

        seeds = [42, 52]  # , 62, 72, 100]

        for seed in seeds:
            logger.info(f"Running training with seed = {seed}")

            set_seed(seed)

            accelerator.wait_for_everyone()

            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

            train_dataloader, eval_dataloader, label_list = build_dataset(
                data_path, tokenizer, config.batch_size
            )
            #         network = build_network(config.fc_layer_size, config.dropout)
            model = model_init(model_name_or_path, label_list, config.lam)
            num_update_steps_per_epoch = len(train_dataloader)
            max_train_steps = epochs * num_update_steps_per_epoch
            optimizer, lr_scheduler = build_optimizer(
                model, config.learning_rate, max_train_steps
            )

            # Prepare everything with our `accelerator`.
            (
                model,
                optimizer,
                train_dataloader,
                eval_dataloader,
                lr_scheduler,
            ) = accelerator.prepare(
                model,
                optimizer,
                train_dataloader,
                eval_dataloader,
                lr_scheduler,
            )

            progress_bar = tqdm(
                range(max_train_steps), disable=not accelerator.is_local_main_process
            )
            completed_steps = 0

            best_eval_f1 = float("inf")

            for epoch in range(epochs):
                for step, batch in enumerate(train_dataloader):
                    outputs = model(**batch)
                    loss = outputs[0]
                    accelerator.backward(loss)

                    if step == len(train_dataloader) - 1:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1

                model.eval()
                eval_loss = 0.0
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)
                    predictions = outputs[1].argmax(dim=-1)
                    predictions, references = accelerator.gather(
                        (predictions, batch["labels"])
                    )
                    eval_loss += outputs[0].item()

                    if accelerator.num_processes > 1:
                        if step == len(eval_dataloader) - 1:
                            predictions = predictions[
                                : len(eval_dataloader.dataset) - samples_seen
                            ]
                            references = references[
                                : len(eval_dataloader.dataset) - samples_seen
                            ]
                        else:
                            samples_seen += references.shape[0]
                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )

                eval_loss = eval_loss / len(eval_dataloader)
                logger.info(f"epoch {epoch}: eval loss: {eval_loss}")

                eval_metric = metric.compute(average="macro")
                eval_f1 = eval_metric["f1"]

                if eval_f1 > best_eval_f1:
                    best_eval_f1 = eval_f1

            results[seed] = best_eval_f1

        avg_f1 = sum(results[k] for k in results) / len(results)
        wandb.log({"eval_f1": avg_f1})


if __name__ == "__main__":
    epochs = 5
    data_path = "data/sample/"
    model_name_or_path = "bert-base-uncased"

    accelerator = Accelerator()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    wandb.login()

    # method
    sweep_config = {"method": "grid"}

    # metric
    sweep_metric = {"name": "eval_f1", "goal": "maximize"}

    # hyperparameters
    parameters_dict = {
        "batch_size": {"values": [2, 4]},
        "learning_rate": {"values": [1e-5, 2e-5]},
        "lam": {"values": [0.1, 0.2]},
    }

    sweep_config["parameters"] = parameters_dict
    sweep_config["metric"] = sweep_metric

    sweep_id = wandb.sweep(sweep_config, project="nairaland-hmc")

    print(sweep_id)

    # wandb.agent(sweep_id, train, count=5)
