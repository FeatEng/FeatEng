from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import VerificationMode, load_dataset, Dataset
from evalplus.eval import FAIL, PASS, TIMEOUT
from evalplus.eval.utils import TimeoutException, create_tempdir, swallow_io, time_limit
from sklearn.metrics import accuracy_score, mean_absolute_error

from feateng.xgboost_model import DataSplit, XGBoostModel

TARGET_COLUMN = "__FeatEng_target__"
HFDS_INDEX_COLUMN = "__index_level_0__"

import resource


def memory_limit():
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (900 * 1024 * 1024 * 1024, hard))
    _, hard = resource.getrlimit(resource.RLIMIT_DATA)
    resource.setrlimit(resource.RLIMIT_DATA, (900 * 1024 * 1024 * 1024, hard))


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


def error_rate_normalizer_mae(baseline_score: float, predicted_score: float) -> float:
    error_rate_reduction = 1.0 - predicted_score / (baseline_score + 1e-16)
    return max(error_rate_reduction, 0)


def error_rate_normalizer_acc(baseline_score: float, predicted_score: float) -> float:
    baseline_error_rate = 1.0 - baseline_score
    predicted_error_rate = 1.0 - predicted_score
    error_rate_reduction = (baseline_error_rate - predicted_error_rate) / (
        baseline_error_rate + 1e-16
    )
    return max(error_rate_reduction, 0)


def unsafe_execute(
    check_program: str,
    result: List[str],
    timeout: float,
    exec_globals: Optional[Dict[str, Any]] = None,
):
    if exec_globals is None:
        exec_globals = {}

    memory_limit()
    try:
        with create_tempdir():
            with swallow_io():
                with time_limit(timeout):
                    exec(check_program, exec_globals)
            result.append(PASS)
    except TimeoutException:
        result.append(TIMEOUT)
    except MemoryError:
        result.append('out_of_memory')
    except BaseException:
        result.append(FAIL)


def split_to_x_and_y(split) -> Tuple[pd.DataFrame, pd.Series]:
    split = split.to_pandas().copy(deep=True)
    y = split[TARGET_COLUMN]
    x = split.drop(columns={TARGET_COLUMN, HFDS_INDEX_COLUMN}, errors="ignore")
    x = x.fillna(value=np.nan)
    return x, y


def process_dataframe_with_code(
    code: str,
    test_dataframe: pd.DataFrame,
    train_dataframe: pd.DataFrame,
    train_target: pd.Series,
    test_target: pd.Series,
) -> Tuple[str, DataSplit]:
    exec_globals = {
        "test_x": test_dataframe,
        "train_x": train_dataframe,
        "train_target": train_target,
    }
    result = []
    unsafe_execute(
        check_program=code,
        result=result,
        timeout=600,
        exec_globals=exec_globals,
    )
    data_split = DataSplit(
        test_target=test_target,
        train_x=exec_globals["train_x"],
        train_target=exec_globals["train_target"],
        test_x=exec_globals["test_x"],
    )
    return result.pop(), data_split

def check_execution_score(
    completion_id: int,
    problem: Dict[str, Any],
    solution: str,
    identifier=None,
    dataset: Optional[Dataset] = None,
):
    try:
        with time_limit(1000):
            return _check_execution_score(completion_id, problem, solution, identifier, dataset)
    except:
        return {
            "completion_id": completion_id,
            "dataframe_id": problem["dataframe_id"],
            "_identifier": identifier,
            "solution": solution,
            "raw_score": 0,
            "benchmark_score": 0,
            "result": "outer_timeout",
        }

def _check_execution_score(
    completion_id: int,
    problem: Dict[str, Any],
    solution: str,
    identifier=None,
    dataset: Optional[Dataset] = None,
) -> Dict[str, Union[str, float]]:  # {...}, "base" | "plus" -> (status, details)
    code = f"import numpy as np\nimport pandas as pd\n{solution}\ntrain_x, train_target, test_x = transform(train_x, train_target, test_x)"

    if dataset is None:
        dataset = load_dataset(
            "FeatEng/Data",
            problem["dataframe_id"],
            verification_mode=VerificationMode.NO_CHECKS,
            keep_in_memory=True,
        )
    train_dataframe, train_target = split_to_x_and_y(dataset["train"])
    test_dataframe, test_target = split_to_x_and_y(dataset["test"])

    result, data_split = process_dataframe_with_code(
        code, test_dataframe, train_dataframe, train_target, test_target
    )

    is_regression = problem["task_type"] == "regression"
    model = XGBoostModel(enable_categorical=True, is_regression=is_regression)
    data_split = model.baseline_encode(data_split)

    try:
        with time_limit(600):
            model.fit(data_split.train_x, data_split.train_target)
            predictions = model.predict(data_split.test_x)

        benchmark_metric = (
            error_rate_normalizer_mae if is_regression else error_rate_normalizer_acc
        )

        if not is_regression:
            raw_score = accuracy_score(
                data_split.test_target.astype("category").cat.codes,
                np.round(predictions),
            )
        else:
            raw_score = mean_absolute_error(data_split.test_target, predictions)

        benchmark_score = benchmark_metric(problem["baseline_score"], raw_score)
    except BaseException:
        raw_score = 0 if not is_regression else float("inf")
        benchmark_score = 0

    return {
        "completion_id": completion_id,
        "dataframe_id": problem["dataframe_id"],
        "_identifier": identifier,
        "solution": solution,
        "raw_score": raw_score,
        "benchmark_score": benchmark_score,
        "result": result,
    }
