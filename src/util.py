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

import sys
import logging
import time
import random
import torch
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger.setLevel(logging.DEBUG)


def count_parameters(m, verbose=True):
    total_count = 0
    learnable_count = 0
    if verbose:
        logger.debug("Parameters (name, tunable, count):")

    output_width = max([len(n) for n, _ in m.named_parameters()])
    for n, p in m.named_parameters():
        count = p.data.numel()
        if verbose:
            logger.debug(f" {n:{output_width}} {p.requires_grad:5b} {count:>11d}")
        total_count += count
        if p.requires_grad:
            learnable_count += count

    logger.info(
        f"Total parameters: {total_count:,}, "
        f"thereof learnable: {learnable_count:,} "
        f"({learnable_count/total_count*100.:5.4f}%)"
    )

    return total_count, learnable_count


def evaluate_tasks(
    inference_pipeline, request_data, task_aware=True, outer_bs=256, inner_bs=8
):
    predictions = multi_task_predict(
        inference_pipeline,
        request_data,
        task_aware,
        outer_bs=outer_bs,
        inner_bs=inner_bs,
    )

    tasks, y = zip(*[(d[0], d[1]) for d in request_data])
    df = pd.DataFrame(data={"y": y, "prediction": predictions, "task": tasks})
    df["correct"] = df.prediction == df.y

    print(
        df.groupby("task").agg(
            {
                "correct": ["mean", "sum"],
                "task": "count",
                "prediction": "sum",
                "y": "sum",
            }
        )
    )
    print("Overall acc:", df["correct"].mean())

    return df


def multi_task_predict(
    inference_pipeline, request_data, task_aware=True, outer_bs=256, inner_bs=8
):
    all_predictions = []
    for i in range(0, len(request_data), outer_bs):
        print(".", end="")
        batch = request_data[i : i + outer_bs]
        task_codes, _, X = zip(*batch)
        predictions = inference_pipeline(
            inputs=X, tasks=task_codes if task_aware else None, bs=inner_bs
        )
        predictions = [pred["label"] for pred in predictions]
        all_predictions.extend(predictions)

    return all_predictions


def execute_with_retry(f):
    return_val = None
    attempt = 0
    while True:
        try:
            return_val = f()
            break
        except EnvironmentError as e:
            if attempt >= 4:
                logger.error(f"Giving up on: {e}")
                logger.exception(e)
                raise e
            attempt += 1
            delay = 4 + random.randint(0, 2**attempt)
            logger.error(
                f"Exception '{e}' during loading of artifacts from hf. Will retry after {delay} seconds."
            )
            time.sleep(delay)

    if return_val is None:
        raise RuntimeError(
            f"Not able to download artifacts from hf with {attempt} attempts."
        )

    return return_val


def calc_combinations(hp_ranges):
    """Ignores non-discrete values"""
    nc = 0

    for hp_name in hp_ranges:
        hpr = hp_ranges[hp_name]
        if hasattr(hpr, "values"):
            if nc == 0:
                nc = 1
            nc *= len(hpr.values)
    return nc


count = 0


def log_gpu_usage(empty_cache=False):
    if not torch.cuda.is_available():
        return

    if empty_cache:
        torch.cuda.empty_cache()

    import pynvml
    from pynvml.smi import nvidia_smi

    nvidia_smi.getInstance()
    device_count = pynvml.nvmlDeviceGetCount()

    global count

    # print general information about the installed GPUs once
    if count == 0:
        logger.info(f"Device count: {device_count}")
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            logger.info(f"Device: {pynvml.nvmlDeviceGetName(handle)}:")
            logger.info(pynvml.nvmlDeviceGetMemoryInfo(handle))

    count += 1

    # print GPU usage on every call
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        relative_used = float(info.used) / float(info.total) * 100
        MB = 1024 * 1024

        logger.info(
            f"GPU Usage. GPU {i} ({count:5d}) Used: {int(info.used)/MB:6.1f} "
            f"MB Total: {int(info.total)/MB:6.1f} MB "
            f"({relative_used:3.1f}% used). Free: {int(info.free)/MB:6.1f} MB"
        )
