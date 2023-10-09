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

import re
import math
import logging
from pathlib import Path
import json
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LoRAAdapter(nn.Module):
    def __init__(self, adaptee, r, dropout=0.1):
        super().__init__()

        self.r = r
        self.dropout = dropout
        self.adaptee = adaptee

        # Store a pointer to the original forward implementation
        # of the module to be adapted.
        # Then point its forward method to this adapter module, effectively
        # injecting our new forward method into the forward path of the
        # encompassing module.
        self.orig_forward = adaptee.forward
        adaptee.forward = self.forward

        # Adding the weight matrices directly to the adaptee,
        # which makes is more practical to report the parameters, as they
        # become part of the overall model.
        adaptee.lora_A = nn.Parameter(
            torch.randn(adaptee.in_features, r) / math.sqrt(adaptee.in_features)
        )
        adaptee.lora_B = nn.Parameter(torch.zeros(r, adaptee.out_features))

    def forward(self, x, *args, **kwargs):
        return (
            self.orig_forward(x, *args, **kwargs)
            + F.dropout(x, self.dropout) @ self.adaptee.lora_A @ self.adaptee.lora_B
        )

    def extra_repr(self):
        return f"LoRAAdapter (r={self.r}, dropout={self.dropout})"


def _generate_adaptable_modules(model, lora_config, default_r):
    """Iterates over all modules and yields those that match a specified regex"""
    # Lookup by name a potentially pre-defined lora_config.
    if lora_config in lora_configs:
        lora_config = lora_configs[lora_config].strip().replace("\n", "|")

    # For all linear modules ...
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # ... check if we have a matching regex for the name of the current
        # module, then extract r and yield.
        for regex in lora_config.split("|"):
            r = re.match(".*\\((\\d+)\\)", regex)
            r = int(r.group(1)) if r else default_r

            # Remove r config from config expression
            regex = re.sub("\\(\\d+\\)", "", regex)
            print("name", name, "regex", regex)

            if re.match(f".*{regex}", name):
                print("matched")
                yield module, name, r
                break


def _cs(p):
    return p.data.sum().item()


def checksum_module(module):
    return {name: _cs(p) for name, p in module.named_parameters()}


def save_adapters(model, clf_module, task_code, model_dir):
    logger.info(f"Saving adapters for task_code {task_code}.")
    save_dir = Path(model_dir) / "tasks" / task_code

    # Which modules are adapted? Let's simply filter for the
    # modules that have an attribute lora_A
    for name, module in model.named_modules():
        module_dir = save_dir / name

        if module == clf_module:
            logger.info(f"Saving classifier {name}.")
            module_dir.mkdir(parents=True, exist_ok=True)
            torch.save(module, module_dir / "clf")
            continue

        if not hasattr(module, "lora_A"):
            continue

        logger.info(f"Saving adapter for {name} to {module_dir}.")
        module_dir.mkdir(parents=True, exist_ok=True)

        # Why do we bake in/merge the weights, as we get rid of the model
        # anyway? Because we want to checksum the adapted modules before
        # saving, so that we can check the validity when installing the
        # adapters at inference time.
        module.weight.data += (module.lora_A @ module.lora_B).transpose(-2, -1)
        torch.save(module.lora_A, module_dir / "lora_A")
        torch.save(module.lora_B, module_dir / "lora_B")

        delattr(module, "lora_A")
        delattr(module, "lora_B")

    # Calculate size of all module adapters for this task
    size = 0
    for f in save_dir.glob("**/*"):
        if not f.is_dir():
            size += f.stat().st_size

    logger.info(f"Size of adapter: {size/1024/1024:7.4f} MB (Excluding checksum.json).")
    with open(save_dir / "checksums.json", "w") as f:
        json.dump(checksum_module(model), f)


def load_adapters(adapters_dir, device):
    adapters = {}
    for task_dir in adapters_dir.iterdir():
        modules = {}
        logger.info(f"task_dir {task_dir}")

        for module_dir in task_dir.iterdir():
            if module_dir.name == "checksums.json":
                continue

            components = {}
            if (module_dir / "lora_A").exists():
                components["lora_A"] = torch.load(
                    module_dir / "lora_A", map_location=torch.device(device)
                )
                components["lora_B"] = torch.load(
                    module_dir / "lora_B", map_location=torch.device(device)
                )
            else:
                components["clf"] = torch.load(
                    module_dir / "clf", map_location=torch.device(device)
                )
            modules[module_dir.name] = components

        adapters[task_dir.name] = modules

        with open(task_dir / "checksums.json", "r") as f:
            checksums = json.load(f)
            assert checksums, "Checksums need to be loaded, but none were."
            adapters[task_dir.name]["checksums"] = checksums

    return adapters


def validate_checksums(model, checksums):
    """Development level tool to validate the mechanical compatibility
    between saved and restored model including the original modules,
    the clf and the adapters."""
    file_keys = set(checksums.keys())
    model_keys = set([name for name, _ in model.named_parameters()])

    extra_model_keys = model_keys - file_keys
    extra_file_keys = file_keys - model_keys

    assert not (extra_file_keys or extra_model_keys), (
        "Need to have the same keys in the model and "
        "in the file checksums.\n"
        f"Extra model keys: {extra_model_keys}, "
        f"extra file keys: {extra_file_keys}"
    )

    errors = []
    for name, p in model.named_parameters():
        cs = _cs(p)
        if not np.allclose(checksums[name], cs, 1e-3):
            errors.append(f"Name: {name} Checksum: {checksums[name]}, p: {cs}")

    if errors:
        logger.error(f"Checksum errors ({len(errors)})\n", "\n".join(errors))
    assert not errors, "Checksums do not match."

    logger.info(
        f"Validated checksums, {len(checksums.keys())} modules, "
        "took {time.time()-started:5.4f}s."
    )
    return len(errors) == 0


def install_adapter(model, adapter, validate=True):
    for mod_name, module in model.named_modules():
        if mod_name in adapter:
            if "lora_A" in adapter[mod_name]:
                lora_AB = adapter[mod_name]["lora_A"] @ adapter[mod_name]["lora_B"]
                # FIXME: Why transpose?
                module.weight.data += lora_AB.transpose(-2, -1)
                logger.debug(f"Installed module adapter for {mod_name}.")

    assert "clf" in adapter["classifier"]
    model.classifier = adapter["classifier"]["clf"]
    logger.debug("Installed clf.")
    if validate:
        checksum_match = validate_checksums(model, adapter["checksums"])
        logger.info(f"Checksums match: {checksum_match}")


# FIXME: Refactor together with the install_adapter to have one method?
def remove_adapter(model, adapter):
    for mod_name, module in model.named_modules():
        if mod_name in adapter:
            if "lora_A" in adapter[mod_name]:
                lora_AB = adapter[mod_name]["lora_A"] @ adapter[mod_name]["lora_B"]
                # FIXME: Why transpose?
                module.weight.data -= lora_AB.transpose(-2, -1)
                logger.debug(f"Removed module adapter for {mod_name}.")
            else:
                # model.classifier = None
                # delattr(model, 'classifier')
                del model.classifier
                logger.debug(f"Removed classifier {mod_name}.")


def add_adapters(model, lora_config, default_r, dropout):
    # freeze the whole model, then unfreeze the clf head
    for p in list(model.children()):
        p.requires_grad_(False)
    for p in list(model.children())[-1].parameters():
        p.requires_grad_(True)

    # Adapt the configured modules.
    # The adapters are new and default to be unfrozen
    for module, name, r in _generate_adaptable_modules(model, lora_config, default_r):
        before_count = module.weight.numel()
        LoRAAdapter(module, r, dropout=dropout)
        after_count = module.lora_A.numel() + module.lora_B.numel()
        logger.debug(
            f"Adapting {name} with r {r}, "
            f"from {before_count} tunable parameters "
            f"to {after_count} ({after_count/before_count:5.4f}%)."
        )

    return model


def _enumerate(lowest_layer_idx, highest_layer_idx, steps):
    """returns upper and lower value and interpolated, evenly spread out, values in between"""
    return "|".join(
        [
            "\." + str(round(i.item())) + "\."
            for i in torch.linspace(lowest_layer_idx, highest_layer_idx, steps)
        ]
    )


lora_configs = {
    "clf_only": "-",
    "all_linears": "roberta.*",
    "all": "query|key|value|attention.output|intermediate|layer.\\d+.output",
    # Vertical/By-Component
    "att_q": "query",
    "att_k": "key",
    "att_v": "value",
    "att_o": "attention.output",
    "ff_u": "intermediate",
    "ff_d": "layer.\\d+.output",
    # Horizontal/By-Layer
    "12_upper": _enumerate(6, 11, 4),
    "12_lower": _enumerate(0, 6, 4),
    "12_even": _enumerate(0, 11, 4),
    "24_upper": _enumerate(12, 23, 6),
    "24_lower": _enumerate(0, 11, 6),
    "24_even": _enumerate(0, 23, 6),
    # Combinations
    "ff_d+ff_u": "layer.\\d+.output|intermediate",
    "att_qk+ff_u": "query|key|intermediate",
    "att_qv+ff_u": "query|value|intermediate",
    "att_q+ff_u": "query|intermediate",
    "att_k+ff_u": "key|intermediate",
    "att_v+ff_u": "value|intermediate",
    "att_o+ff_u": "attention.output|intermediate",
    # short vs long path to error
    "12_top2_ff_u": "1[01].*?intermediate",
    "12_lowest2_ff_u": "\\.[01]\\..*?intermediate",
}
