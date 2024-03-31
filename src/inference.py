# MIT No Attribution
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from pathlib import Path
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import set_seed
from sagemaker_inference.decoder import decode
from sagemaker_inference.encoder import encode

import numpy as np
import torch

from lora import load_adapters, install_adapter, remove_adapter

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiTaskPipeline:
    def __init__(self, model, tokenizer, validate_adapters=False, task_adapters=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.task_adapters = task_adapters if task_adapters else {}
        self.validate_adapters = validate_adapters
        self.current_task = None
        logger.info(f"MultiTaskPipeline init(). ({self}, {id(self)})")

    def __call__(self, inputs, tasks=None, bs=8):
        # Not in task mode?
        if not tasks or self.task_adapters is None:
            # Called in task mode, likely accidentally, but no adapters were loaded?
            assert not self.current_task, (
                "Can't call inference without a task "
                "on a model that is setup for a task."
            )
            # No inner batch handling here. We just use the outer batch, as we
            # do not group inference calls by task, given that we have no
            # adapters, so there can only be one task.
            tokenized = self.tokenizer(inputs, padding=True, truncation=True)
            return self._predict(
                zip(
                    list(range(len(tokenized.input_ids))),
                    tokenized.input_ids,
                    tokenized.attention_mask,
                )
            )
        # If tasks are specified then the there must be one task per input
        assert len(tasks) == len(
            inputs
        ), "When specifiying tasks, one task per input needed."

        # We ensured that we have one task per input smaple,
        # but do we have adapters loaded for all tasks occurring?
        all_tasks = set(tasks)
        for task in all_tasks:
            assert task in self.task_adapters, (
                f"Task {task} specified in inputs,"
                "but no adapters for that task loaded."
            )

        results = []

        # Inference grouped by task,
        # so that we minimize the # of adapter changes
        for task in all_tasks:
            logger.info(f"Running inference for task: {task}")
            self._change_adapter(new_task=task)

            # Order by task:
            # We can process the requests in any order we want, as we only
            # return results to the client once, i.e. not streaming.
            # We use that to our advantage and group the processing by task.
            # This reduces the number of task adapter changes necessary.

            # Preserve original order of input samples:
            # We introduce synthetic ids, task_input_ids, to keep track of
            # the original position of the requests in the input.
            # So that at the end we can return the results in the original
            # order of the request, so that the caller can rely on order.

            # FUTURE: Continue using the already installed adapter:
            # It would complicate the code a little bit, but starting with
            # the task that is potentially already loaded from the last
            # invocation would reduce the number of adapter changes. As the
            # order of execution does not impact the order of results, this
            # optimization is desirable.
            task_input_ids = [i for i in range(len(inputs)) if tasks[i] == task]
            for i in range(0, len(task_input_ids), bs):
                batch_task_input_ids = task_input_ids[i : i + bs]
                inp = np.array(inputs)[[batch_task_input_ids]].tolist()
                tokenized = self.tokenizer(
                    inp,
                    padding=True,
                    truncation=True,
                )
                task_inputs = zip(
                    batch_task_input_ids, tokenized.input_ids, tokenized.attention_mask
                )

                results.extend(self._predict(task_inputs))

        logger.info(f"Inference call returns {len(results)} predictions.")
        return sorted(results, key=lambda _: _["id"])

    def _change_adapter(self, new_task):
        started_ns = time.time_ns()
        assert (
            new_task
        ), "Changing adapters makes only sense when specifying a new task."

        # Same task as before -> done, noop
        if new_task == self.current_task:
            return

        # Adapter installed? Then remove it before installing a new one.
        previous_task = self.current_task
        if self.current_task:
            logger.info(f"Removing adapter {self.current_task}.")
            remove_adapter(self.model, self.task_adapters[self.current_task])
            self.current_task = None

        # At this point we have a new task,
        # there was no existng adapter, i.e. freshly loaded pre-trained model,
        # or we already removed the adapter.
        # Adding the adapter for our new_task now.
        logger.info(f"Installing adapter {new_task}.")

        install_adapter(
            self.model, self.task_adapters[new_task], self.validate_adapters
        )
        self.current_task = new_task
        self.model.eval()

        duration_ms = (time.time_ns() - started_ns) / 1e6
        logger.info(f"Changing adapter (remove and install) from {previous_task} to {new_task} took {duration_ms:5.2f} ms.")
        return 

    def _predict(self, inputs):
        sample_ids, input_ids, attention_masks = zip(*inputs)

        input_ids, attention_masks = torch.tensor(input_ids, device=dev), torch.tensor(
            attention_masks, device=dev
        )

        with torch.inference_mode():
            logits = self.model(input_ids, attention_masks).logits.cpu().detach()

        # FIXME: Is detach() still necessary?
        cls_idx = logits.argmax(-1).detach().numpy()
        scores = logits.softmax(-1).detach()
        return [
            {
                "id": sample_id,
                "logit": logits[idx].item(),
                "score": scores[idx].item(),
                "label": idx.item(),
            }
            for sample_id, logits, scores, idx in zip(
                sample_ids, logits, scores, cls_idx
            )
        ]


def model_fn(model_dir):
    seed = 42
    if seed:
        logger.info(f"Setting seed to: {seed}")
        set_seed(seed)

    started = time.time()
    logger.info(f"Calling model_fn with model_dir: {model_dir}")

    if logger.level == logging.DEBUG:
        logger.debug("files in model_dir:")
        for f in Path("/opt/ml").glob("**/*"):
            logger.debug(f)

    logger.debug("Loading model and further artefacts.")

    # model
    model_dir = Path(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir / "model")
    model = model.to(dev)
    model.eval()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")

    # adapters
    adapters_dir = Path(model_dir) / "tasks"
    logger.info(f"Found adapters: {adapters_dir.exists()}")

    adapters = None

    if adapters_dir.exists():
        adapters = load_adapters(adapters_dir, dev)
        logger.info(f"Loaded adapters {adapters.keys()}.")

    # pipeline
    inference_pipeline = MultiTaskPipeline(
        model, tokenizer, task_adapters=adapters, validate_adapters=False
    )

    logger.info(f"Loaded pipeline {id(inference_pipeline)} with model {id(model)}.")
    logger.info(f"loading took {time.time()-started:4.2f}s.")

    return inference_pipeline


def transform_fn(model, input_data, content_type, accept):
    data_decoded = decode(input_data, content_type).item()
    inputs = data_decoded["inputs"]
    bs = (
        data_decoded["parameters"]["batch-size"]
        if "parameters" in data_decoded and "batch-size" in data_decoded["parameters"]
        else 8  # Default batch size
    )
    tasks = data_decoded.get("tasks", None)
    result = model(inputs, tasks=tasks, bs=bs)
    result_encoded = encode(result, accept)

    return result_encoded
