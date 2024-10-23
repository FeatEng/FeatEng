import json
import os
import re
from typing import List, Optional

from evalplus.provider import DecoderBase
from evalplus.utils import progress

from feateng.data import get_feateng
from feateng.provider import make_model


def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    try:
        return re.findall(r"```python\n([^`]+)(```)?", code, re.MULTILINE | re.DOTALL)[
            -1
        ][0]
    except:
        return ""


def codegen(
    target_path: str,
    model: DecoderBase,
    dataset: str,
    greedy=False,
    n_samples=1,
    id_range=None,
    resume=True,
):
    task2nexist = {}
    if resume and target_path.endswith(".jsonl") and os.path.isfile(target_path):
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                task_id = json.loads(line)["task_id"]
                task2nexist[task_id] = task2nexist.get(task_id, 0) + 1

    if target_path.endswith(".jsonl"):
        raw_target_path = target_path.replace(".jsonl", ".raw.jsonl")
    else:
        raw_target_path = target_path + ".raw"
        os.makedirs(target_path, exist_ok=True)

    print(f"Sanitized code outputs will be saved to {target_path}")
    print(f"Raw outputs will be saved to {raw_target_path}")

    with progress(dataset) as p:
        if dataset == "feateng":
            dataset = get_feateng()
        else:
            raise ValueError(f"Invalid dataset {dataset}")

        for task_id, task in p.track(dataset.items()):
            if id_range is not None:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            if not target_path.endswith(".jsonl"):
                p_name = task_id.replace("/", "_")
                os.makedirs(os.path.join(target_path, p_name), exist_ok=True)
                task2nexist[task_id] = len(
                    [
                        f
                        for f in os.listdir(os.path.join(target_path, p_name))
                        if f.endswith(".py")
                    ]
                )

            n_more_samples = n_samples
            log = f"Codegen: {task_id} @ {model}"
            if resume and task2nexist.get(task_id, 0) > 0:
                log += f" (resuming from {task2nexist[task_id]})"
                n_more_samples -= task2nexist[task_id]

            p.console.print(log)

            sidx = n_samples - n_more_samples
            while sidx < n_samples:
                prompt = task["prompt"].strip() + "\n"
                outputs = model.codegen(
                    prompt,
                    do_sample=not greedy,
                    num_samples=n_samples - sidx,
                )
                assert outputs, "No outputs from model!"
                for impl in outputs:
                    solution = prompt + impl if model.is_direct_completion() else impl
                    sanitized_solution = sanitize(solution, entrypoint=None)
                    if target_path.endswith(".jsonl"):
                        # Writing the sanitized version
                        with open(target_path, "a") as f:
                            f.write(
                                json.dumps(
                                    {"task_id": task_id, "solution": sanitized_solution}
                                )
                                + "\n"
                            )

                        # Writing the raw version
                        with open(raw_target_path, "a") as f:
                            f.write(
                                json.dumps({"task_id": task_id, "solution": solution})
                                + "\n"
                            )
                    else:
                        # Writing the sanitized version
                        with open(
                            os.path.join(target_path, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(sanitized_solution)

                        # Writing the raw version
                        with open(
                            os.path.join(raw_target_path, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(solution)
                    sidx += 1


def run_codegen(
    model: str,
    dataset: str = "feateng",
    root: str = "results",
    bs: Optional[int] = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    resume: bool = True,
    greedy: bool = False,
    id_range: List = None,
    version: str = "default",
    backend: str = "vllm",
    force_base_prompt: bool = False,
    base_url: str = None,
    tp: int = 1,
    evalperf_type: str = None,  # For EvalPerf
    jsonl_fmt: bool = True,
    attn_implementation: str = "eager",
    trust_remote_code: bool = False,
    dtype: str = "bfloat16",
):
    assert dataset in ["feateng"], f"Invalid dataset {dataset}"

    if greedy and (temperature != 0 or bs != 1 or n_samples != 1):
        temperature = 0.0
        bs = 1
        n_samples = 1
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if id_range is not None:
        assert len(id_range) == 2, "id_range must be a list of length 2"
        assert id_range[0] < id_range[1], "id_range must be increasing"
        id_range = tuple(id_range)

    if bs is None:
        bs = min(n_samples, 32)
        print(f"Setting batch size to {bs}")

    # Make project dir
    os.makedirs(root, exist_ok=True)
    # Make dataset dir
    os.makedirs(os.path.join(root, dataset), exist_ok=True)

    # Model creation
    model_runner = make_model(
        model=model,
        backend=backend,
        batch_size=bs,
        temperature=temperature,
        force_base_prompt=force_base_prompt,
        dataset=dataset,
        base_url=base_url,
        tp=tp,
        instruction_prefix="",
        response_prefix="",
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )

    # Make dir for codes generated by each model
    identifier = model.strip("./").replace("/", "--") + f"_{backend}_temp_{temperature}"
    if evalperf_type:
        identifier += f"-{evalperf_type}"

    target_path = os.path.join(root, dataset, identifier)
    if jsonl_fmt:
        target_path += ".jsonl"
    else:
        os.makedirs(target_path, exist_ok=True)
    codegen(
        target_path=target_path,
        dataset=dataset,
        greedy=greedy,
        model=model_runner,
        n_samples=n_samples,
        resume=resume,
        id_range=id_range,
    )

    # force shutdown the model runner
    del model_runner
    import gc

    gc.collect()

    return target_path


def main():
    from fire import Fire

    Fire(run_codegen)


if __name__ == "__main__":
    main()
