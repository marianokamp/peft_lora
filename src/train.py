# MIT No Attribution
#
# Copyright Mariano Kamp
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import os
import sys
import logging

import time

from pathlib import Path
from functools import partial

from threading import Thread

import random
import numpy as np

import torch

import transformers
from transformers import (
    DataCollatorWithPadding,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from transformers import set_seed
import datasets
from datasets import load_dataset
import evaluate

from util import count_parameters, execute_with_retry, log_gpu_usage
from lora import add_adapters, save_adapters

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger.setLevel(logging.INFO)
logger.info(f"torch version: {torch.__version__}")

transformers.logging.set_verbosity_warning()

tasks = ["sst2", "cola"]


def schedule_gpu_memory_logging():
    def log_loop():
        while True:
            log_gpu_usage()
            time.sleep(30)

    t = Thread(target=log_loop, daemon=True)
    t.start()


if __name__ == "__main__":
    schedule_gpu_memory_logging()


def load_and_prepare_data(
    task, tokenizer, n_train_samples=None, n_valid_samples=None, scale_input=1
):
    logger.info(
        f"lapd called with n_train: {n_train_samples}, n_valid: {n_valid_samples}"
    )
    datasets.logging.disable_progress_bar()
    dataset = load_dataset("glue", task)

    def preprocess_function(examples):
        # We are doing the padding in the collator, to make sure that we
        # do align the necessary padding length with the surrounding
        # mini-batch.
        return tokenizer(
            scale_input * examples["sentence"], padding=False, truncation=True
        )

    train = dataset["train"]
    valid = dataset["validation"]

    if n_train_samples:
        train = train.shuffle(seed=42).select(
            [i for i in list(range(min(n_train_samples, len(train))))]
        )

    if n_valid_samples:
        valid = valid.shuffle(seed=42).select(
            [i for i in list(range(min(n_valid_samples, len(valid))))]
        )

    tokenized_train = train.map(preprocess_function, batched=False)
    tokenized_valid = valid.map(preprocess_function, batched=False)

    mean_length = np.mean([len(v["input_ids"]) for v in tokenized_train])
    logger.info(f"Average train input length: {mean_length:5.2f}")

    return tokenized_train, tokenized_valid


def load_model_and_tokenizer_and_collator(
    model_ckp,
    clf_dropout=0.0,
    hidden_dropout=0.1,
    attention_dropout=0.1,
    model_dir=None,
):
    args = [model_dir if model_dir and (model_dir / "model").exists() else model_ckp]

    model = execute_with_retry(
        lambda: AutoModelForSequenceClassification.from_pretrained(
            *args,
            num_labels=2,
            classifier_dropout=clf_dropout,
            hidden_dropout_prob=hidden_dropout,
            attention_probs_dropout_prob=attention_dropout,
        )
    )
    tokenizer = execute_with_retry(lambda: AutoTokenizer.from_pretrained(model_ckp))
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return model, tokenizer, collator


def prepare_data(
    tasks_to_execute, tokenizer, randomize=False, max_samples_per_task=None
):
    """Util function used during inference."""
    prepared_data = []

    for task in tasks_to_execute:
        logger.info(f"Preparing data for task: {task}")

        _, valid_tokenized = load_and_prepare_data(task, tokenizer)

        Xs = valid_tokenized["sentence"]
        ys = valid_tokenized["label"]
        task_codes = len(valid_tokenized) * [task]

        examples = [(task_code, y, X) for task_code, y, X in zip(task_codes, ys, Xs)]
        prepared_data.extend(examples[:max_samples_per_task])

    # We need the randomness to make sure that we don't
    # only have subsequent calls within one task as we got data in order.
    # We are just shuffling the order in the overall batch,
    # but we still only use the first n examples, we truncated
    # above.
    if randomize:
        logger.info("Randomizing order in input sequence.")
        random.shuffle(prepared_data)
    else:
        logger.info("Not randomizing order in input sequence.")

    return prepared_data


def compute_metrics(task, eval_pred):
    load_matthews_correlation = evaluate.load("matthews_correlation")
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = load_accuracy.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    matthews_correlation = load_matthews_correlation.compute(
        predictions=predictions, references=labels
    )["matthews_correlation"]

    metrics = {
        f"{task}_accuracy": accuracy,
        f"{task}_f1": f1,
        f"{task}_matthews_correlation": matthews_correlation,
    }
    log_gpu_usage(args.empty_cuda_cache)

    return metrics


def add_hf_lora_adapters(model, lora_config, default_r, lora_alpha, lora_dropout):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
    )

    # target modules documentation:
    # https://github.com/huggingface/peft/blob/d6015bc11fc9a194c54ed9337513716ac6554437/src/peft/tuners/lora.py#L192

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=default_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        # Can't differentiate between attention.output and layer.\d+.output? How would you
        # take the up projection, but not the attention output?
        target_modules=[
            "query",
            "key",
            "value",
            "output.dense",
            "intermediate.dense",
        ]
        if lora_config.startswith("all")
        else lora_config.split("|"),
        # modules_to_save=["classifier"], # https://github.com/huggingface/peft/blob/032fff92fb74b737a2934e91a08d82142fb79dc3/src/peft/peft_model.py#L644
    )

    model = get_peft_model(model, peft_config)
    return model


def pre_fit_task(args):
    if args.use_hf_logging:
        from transformers.utils import logging_hf

        logging_hf.set_verbosity_info()
        # logger = logging.get_logger("transformers")

    logger.info("Loading model and tokenizer.")
    model, tokenizer, collator = load_model_and_tokenizer_and_collator(
        args.model_ckp, args.clf_dropout, args.hidden_dropout, args.attention_dropout
    )

    logger.info(f"Pre-trained model:\n{model}")

    # Saving the tokenizer will be done for all scenarios.
    logger.info("Saving tokenizer.")
    tokenizer.save_pretrained(Path(args.model_dir) / "tokenizer")

    # In case we are not using hf peft, we will take care of
    # saving all artifacts by ourselves. So here, we would
    # save the pre-trained model. Later we will add the
    # adapters.
    # With hf we will only allow for one task at this time,
    # and we will use the standard way, to save the finetuned
    # model at the end.

    # Save here, if we do not leave it to
    # hf_lora (args.use_hf_lora) *or* we are not training
    # vanilla without any LoRA applied (args.lora_conf)
    # This means we only save the pre-trained model here,
    # if we apply LoRA ourselves.
    # For simplicity, so that we do not have to check all
    # the nested config parameters for different tasks, that
    # may be set but are not active? -> FIXME
    if not args.use_hf_lora:
        logger.info("Saving pre-trained model")
        model.save_pretrained(Path(args.model_dir) / "model")

    return tokenizer, collator


def fit_task(task, args, tokenizer, collator):
    # Not using the generic pre-trained model,
    # from the pre_fit_task method.
    # Specifically instead now instantiating one with
    # the specific number of labels for our task
    model, _, _ = load_model_and_tokenizer_and_collator(
        args.model_ckp,
        args.clf_dropout,
        args.hidden_dropout,
        args.attention_dropout,
        Path(args.model_dir) / "model",
    )  # contains the pre-trained model
    model.train()
    logger.info("Pre-trained model loaded.")

    # Install adapters
    if args.use_hf_lora:
        logger.info("Adapting model using hf peft.")
        model = add_hf_lora_adapters(
            model, args.lora_config, args.lora_r, args.hf_lora_alpha, args.lora_dropout
        )

    elif args.lora_config is not None and args.lora_config.lower() != "none":
        logger.info("Adapting model.")
        model = add_adapters(
            model=model,
            lora_config=args.lora_config,
            default_r=args.lora_r,
            dropout=args.lora_dropout,
        )

    logger.info("Loading and preparing data.")
    tokenized_train, tokenized_valid = load_and_prepare_data(
        task, tokenizer, args.n_train_samples, args.n_valid_samples, args.scale_input
    )
    logging_steps = len(tokenized_train) // args.batch_size // 4 
    logger.info(
        f"Training with lr: {args.learning_rate}, batch-size: {args.batch_size}, wd: {args.weight_decay}, log steps: {logging_steps}."
    )
    training_args = TrainingArguments(
        output_dir=args.model_dir if args.model_dir else "out",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        push_to_hub=False,
        load_best_model_at_end=args.patience,
        fp16=args.use_fp16 and torch.cuda.is_available(),
        bf16=args.use_bf16 and torch.cuda.is_available(),
        warmup_steps=args.n_warmup_steps,
        warmup_ratio=args.warmup_ratio,
        adam_epsilon=1e-6,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        # save_steps=1,
        logging_steps=logging_steps,
        # metric_for_best_model=f"eval_{task}_accuracy",
        gradient_checkpointing=args.use_gradient_checkpointing,
        gradient_accumulation_steps=args.n_gradient_accumulation_steps,
        logging_strategy="steps",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        seed=args.seed,
        disable_tqdm=True,
        # logging_dir="/opt/ml/output/tensorboard",
        # report_to=["tensorboard"],
        lr_scheduler_type="cosine",  # cosine, linear
    )

    callbacks = []

    if args.patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=partial(compute_metrics, task),
        callbacks=callbacks,
    )

    total_parameters, learnable_parameters = count_parameters(model)
    logger.info(
        f"total_parameters: {total_parameters} learnable_parameters: {learnable_parameters}"
    )

    logger.info("Starting training.")
    trainer.train()
    logger.info("Training done. Starting evaluation.")
    trainer.evaluate()

    logger.info("Evaluation done.")

    if args.lora_config and not args.use_hf_lora:
        # Use the custom LoRA at this point we already have saved
        # the pre-trained model, but need to save the LoRA modules
        save_adapters(model, list(model.children())[-1], task, args.model_dir)

    else:
        # We do not use hf lora, which would save itself, but also did not
        # train a LoRA version ourselves, so we are in the plain vanilla mode
        # and need to save the now finetuned model.
        model.save_pretrained(Path(args.model_dir) / "model")

    logger.info("Saving model done.")
    try:
        logger.info(
            f'Model size: {(Path(args.model_dir)/"model"/"pytorch_model.bin").stat().st_size/1024.0/1024.0:7.4f} MB.'
        )
    except:
        pass

    return


def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.

    # We load the datasets inside of the training job, therefore we do not use
    # the train channel here.
    # parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--model-ckp", type=str, default="roberta-base")

    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--clf-dropout", type=float, default=0.0)
    parser.add_argument("--hidden-dropout", type=float, default=0.1)
    parser.add_argument("--attention-dropout", type=float, default=0.1)

    parser.add_argument("--n-train-samples", type=int, default=0)
    parser.add_argument("--n-valid-samples", type=int, default=0)
    parser.add_argument("--scale-input", type=int, default=1)
    parser.add_argument("--n-warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument(
        "--use-gradient-checkpointing", type=int, default=1, help="1 yes, 0 no"
    )
    parser.add_argument("--n-gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--empty-cuda-cache", type=int, default=1)
    parser.add_argument("--use-mps", type=int, default=0)
    parser.add_argument("--use-bf16", type=int, default=0)
    parser.add_argument("--use-fp16", type=int, default=0)
    parser.add_argument("--use-hf-logging", type=int, default=0)
    parser.add_argument(
        "--use-hf-lora",
        type=int,
        default=0,
        help="1 for hfi peft. Applies to all tasks.",
    )
    parser.add_argument(
        "--tasks", type=str, default="sst2", help="Comma separated list of tasks."
    )

    # per-task hyperparameters
    for task in tasks:
        parser.add_argument(f"--{task}-learning-rate", type=float, default=2e-5)
        parser.add_argument(f"--{task}-weight-decay", type=float, default=1e-2)
        parser.add_argument(f"--{task}-batch-size", type=int, default=16)
        parser.add_argument(f"--{task}-epochs", type=float, default=10.0)
        parser.add_argument(f"--{task}-lora-dropout", type=float, default=0.1)
        parser.add_argument(
            f"--{task}-lora-config",
            type=str,
            default=None,
            help="e.g. attn(4), mlp(6), where the number is the "
            "r for components' whose names match the regexp.",
        )
        parser.add_argument(
            f"--{task}-lora-r",
            type=int,
            default=2,
            help="default r if not specified in --lora-config.",
        )
        parser.add_argument(
            f"--{task}-hf-lora-alpha",
            type=int,
            default=32,
        )

    args, _ = parser.parse_known_args()

    logger.info(f"Arguments: {args}")

    return args


def _task_scoped_args(task, args):
    task_scoped_args = argparse.Namespace(**vars(args))  # copy
    for k, v in vars(args).items():
        if k.startswith(task + "_"):
            delattr(task_scoped_args, k)
            new_k = k.replace(task + "_", "")
            setattr(task_scoped_args, new_k, v)

            # So that we can pass in 'none' and use it as a label.
            if new_k == "lora_config" and v == "none":
                task_scoped_args.lora_config = None

    return task_scoped_args


def fit(args):
    if args.seed:
        logger.info(f"Setting seed to: {args.seed}.")
        set_seed(args.seed)
    else:
        logger.info("No seed set.")

    tokenizer, collator = pre_fit_task(args)

    # Call fit for each task with a scoped version of the args
    # We scope the arguments so that to each individual train loop
    # we can pretend that the task is the only one we execute,
    # and don't need to differentiate between task specific
    # and global args.
    for task in tasks:
        logger.info(f"Fitting task {task}.")
        if task not in args.tasks.split(","):
            logger.info(
                f"Skipping task {task}, as it is not specified in 'args.tasks'."
            )
            continue

        fit_task(task, _task_scoped_args(task, args), tokenizer, collator)


if __name__ == "__main__":
    args = parse_args()
    fit(args)


from inference import model_fn, transform_fn
