import json
import multiprocessing
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional
from warnings import warn

import datasets
import numpy as np
from evalplus.config import *
from evalplus.data import load_solutions
from evalplus.eval import compatible_eval_result
from termcolor import cprint
from tqdm import tqdm

from feateng.codegen import run_codegen
from feateng.data import get_feateng, get_feateng_dataframes_builders, get_feateng_hash
from feateng.score import check_execution_score


def evaluate(
    samples: Optional[str] = None,
    parallel: int = 2,
    i_just_wanna_run: bool = False,
    **model_kwargs,
):
    dataset = "feateng"

    # datasets.logging.set_verbosity_error()
    # datasets.disable_progress_bars()

    if model_kwargs:
        # To suppress the warning of tokenizers
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false"
        )
        samples = run_codegen(
            dataset=dataset,
            **model_kwargs,
        )
    assert samples is not None, "No samples provided"

    n_workers = parallel

    if os.path.isdir(samples):
        result_path = os.path.join(samples, "eval_results.json")
    else:
        assert samples.endswith(".jsonl")
        result_path = samples.replace(".jsonl", "_eval_results.json")

    if os.path.isfile(result_path) and not i_just_wanna_run:
        print(f"Load from previous results from {result_path}")
        with open(result_path, "r") as f:
            results = json.load(f)

        results = compatible_eval_result(results)
    else:
        problems = get_feateng()
        dataset_hash = get_feateng_hash()

        results = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "hash": dataset_hash,
            "eval": {},
        }

        print("Preloading datasets...")
        preloaded_datasets = {
            task_id: builder.as_dataset()
            for builder, task_id in zip(
                get_feateng_dataframes_builders(), problems.keys()
            )
        }

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            eval_results = defaultdict(list)  # task_id ->
            remainings = set()

            print("Reading samples...")
            for sample in tqdm(load_solutions(samples)):
                task_id = sample["task_id"]
                if task_id not in problems:
                    warn(
                        f"Task {task_id} is found in the samples but not found in the dataset"
                    )
                    continue
                solution = (
                    sample["solution"]
                    if "solution" in sample
                    else problems[task_id]["prompt"] + sample["completion"]
                )
                remainings.add(sample["_identifier"])
                args = (
                    completion_id[task_id],
                    problems[task_id],
                    solution,
                    sample["_identifier"],
                    preloaded_datasets[task_id],
                )

                futures.append(executor.submit(check_execution_score, *args))
                completion_id[task_id] += 1
                n_samples += 1

            assert n_samples == len(remainings), "Missing problems in unfinished"
            assert len(completion_id) == len(problems), "Missing problems in samples"

            print("Executing... This may take a while when running for the first time.")

            for future in tqdm(as_completed(futures), total=n_samples):
                result = future.result()
                print(result["_identifier"])
                remainings.remove(result["_identifier"])
                eval_results[result["dataframe_id"]].append(result)

            results["eval"] = eval_results

    total_scores = defaultdict(list)
    for group in results["eval"].values():
        for idx, single in enumerate(group):
            total_scores[idx].append(single["benchmark_score"])
    total_scores = [np.mean(total) for total in total_scores.values()]
    total = np.mean(total_scores)
    std = np.std(total_scores)

    cprint(f"FeatEng (bechmark score)", "green")
    if std > 0:
        cprint(f"{total:.3f} ± {std:.3f}", "green")
    else:
        cprint(f"{total:.3f}", "green")

    # save results
    if os.path.isfile(result_path) and i_just_wanna_run:
        decision = ""
        while decision.lower() not in ["y", "n"]:
            print(f"{result_path} already exists. Press [Y/N] to overwrite or exit...")
            decision = input()

        if decision.lower() == "y":
            # mv the file to a backup
            new_path = result_path + ".bak"
            while os.path.isfile(new_path):
                new_path += ".bak"
            os.rename(result_path, new_path)
            print(f"Backup {result_path} to {new_path}")

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(results, f)


def main():
    from fire import Fire

    Fire(evaluate)


if __name__ == "__main__":
    main()
