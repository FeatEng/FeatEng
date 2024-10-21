from typing import Dict, Iterable

from datasets import (
    DatasetBuilder,
    DownloadMode,
    VerificationMode,
    disable_progress_bars,
    load_dataset,
    load_dataset_builder,
)

BENCHMARK_NAME = "FeatEng/Benchmark"
BENCHAMARK_DATA_NAME = "FeatEng/Data"
BENCHMARK_SPLIT = "test"


def get_feateng() -> Dict[str, Dict]:
    dataset = load_dataset(BENCHMARK_NAME, split=BENCHMARK_SPLIT)
    return {f"feateng/{idx}": item for idx, item in enumerate(dataset)}


def get_feateng_hash() -> str:
    return load_dataset(
        BENCHMARK_NAME,
        split=BENCHMARK_SPLIT,
        verification_mode=VerificationMode.NO_CHECKS,
    )._fingerprint


def get_feateng_dataframes_builders(num_proc: int = 1) -> Iterable[DatasetBuilder]:
    for problem in load_dataset(
        BENCHMARK_NAME,
        split=BENCHMARK_SPLIT,
        verification_mode=VerificationMode.NO_CHECKS,
    ):
        disable_progress_bars()
        builder = load_dataset_builder(
            BENCHAMARK_DATA_NAME,
            problem["dataframe_id"],
            download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
        )
        builder.download_and_prepare(
            verification_mode=VerificationMode.NO_CHECKS,
            num_proc=num_proc,
        )
        yield builder
