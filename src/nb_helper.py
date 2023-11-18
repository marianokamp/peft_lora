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

from pathlib import Path
import toml

import pandas as pd
import boto3
import altair as alt
from IPython.display import display

from transformers import AutoModelForSequenceClassification
from sagemaker.tuner import HyperparameterTuner, IntegerParameter
from sagemaker import get_execution_role

from amtviz import visualize_tuning_job
from util import count_parameters

sm = boto3.client("sagemaker")

m = AutoModelForSequenceClassification.from_pretrained("roberta-base")
roberta_total, roberta_learnable = count_parameters(m, verbose=False)


def get_default_estimator_parameters():
    return p("estimator_parameters") | {
        "role": get_execution_role(),
        "metric_definitions": p("metric_definitions"),
        "hyperparameters": p("hyperparameters"),
    }


def p(key=None):
    d = toml.load(Path("src/parameters.toml"))
    if key:
        return d[key]
    return d


# Lets' keep track of the baselines and the major experiment outcomes
def capture_results(title, template_estimator, source_tuner=None, job_name=None):
    n_trials = 3
    hpt_ranges = {"dummy": IntegerParameter(0, n_trials)}

    # We take the source tuner as a template
    if source_tuner is not None:
        best_hyperparameters = sm.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=source_tuner.describe()[
                "HyperParameterTuningJobName"
            ]
        )["BestTrainingJob"]["TunedHyperParameters"]
        best_hyperparameters = {
            k: float(v.replace('"', "")) for k, v in best_hyperparameters.items()
        }
        template_estimator.set_hyperparameters(**best_hyperparameters)

    tuner_parameters = {
        **(
            p("tuner_parameters")
            | dict(
                strategy="Random",
                hyperparameter_ranges=hpt_ranges,
                metric_definitions=p("metric_definitions"),
                estimator=template_estimator,
                base_tuning_job_name=job_name if job_name else "capture",
                max_jobs=n_trials,
                max_parallel_jobs=n_trials,
            )
        )
    }

    target_tuner = HyperparameterTuner(**tuner_parameters)
    target_tuner.fit()
    target_tuner_name = target_tuner.describe()["HyperParameterTuningJobName"]

    _, trials_df, full_df = visualize_tuning_job(target_tuner, return_dfs=True)
    full_df.sort_values("ts", inplace=True)

    parameters_relative = (
        full_df[full_df.label == "learnable_parameters"]["value"].values[-1]
        / roberta_learnable
        * 100
    )
    train_speed_median = full_df[full_df.label == "train_samples_sec"]["value"].median()
    gpu_memory_max = full_df[full_df.label == "gpu_memory"]["value"].max()
    # gpu_memory_median = full_df[full_df.label == "gpu_memory"]["value"].median()

    objective_metric_mean = trials_df.iloc[:, -1].mean()
    objective_metric_std = trials_df.iloc[:, -1].std()

    # Merge results into overall results
    results_p = Path("results.csv")
    existing_df = pd.read_csv(results_p) if results_p.exists() else pd.DataFrame()

    new_df = pd.DataFrame(
        data={
            "target_tuner_name": [target_tuner_name],
            "title": [title],
            "objective_metric_mean": [objective_metric_mean],
            "objective_metric_std": [objective_metric_std],
            "parameters_relative": [parameters_relative],
            "train_speed_median": [train_speed_median],
            "gpu_memory_max": [gpu_memory_max],
            # "gpu_memory_median": [gpu_memory_median],
        }
    )

    pd.concat([existing_df, new_df]).to_csv(results_p, index=False)


def graph_results():
    df = pd.read_csv(Path("results.csv"))

    performance = (
        alt.Chart(df, title="Model Performance", width=200)
        .mark_bar(color="orange")
        .encode(
            x=alt.X("title:N", sort=None),
            y=alt.Y(
                "objective_metric_mean:Q",
                stack=None,
                scale=alt.Scale(zero=False, padding=50),
            ),
        )
    )
    parameters = (
        alt.Chart(
            df[df.title != "Full Finetuning"],
            title="Parameters (% of Full Finetuning)",
            width=200,
        )
        .mark_bar(color="orange")
        .encode(
            x=alt.X("title:N", sort=None),
            y=alt.Y("parameters_relative:Q", scale=alt.Scale(zero=False)),
        )
    )
    gpu_mem = performance.properties(title="GPU Memory").encode(
        y=alt.Y("gpu_memory_max:Q", scale=alt.Scale(zero=False))
    )
    train_speed = performance.properties(title="Train Velocity").encode(
        y=alt.Y("train_speed_median:Q", scale=alt.Scale(zero=False))
    )
    display(df)
    return performance | parameters | gpu_mem | train_speed


def display_tuning_jobs(tuning_jobs):
    if not isinstance(tuning_jobs, list):
        tuning_jobs = [tuning_jobs]

    for tj in tuning_jobs:
        if not isinstance(tj, str):
            tj.wait()
            tj = tj.describe()["HyperParameterTuningJobName"]
        display(tj)
        display(
            visualize_tuning_job(
                tj,
                job_metrics=[
                    "sst2_valid_acc",
                    "train_loss",
                    "learnable_parameters",
                    "learning_rate",
                    "gpu_memory",
                    "epoch",
                    "train_samples_sec",
                ],
                advanced=True,
            )
        )
