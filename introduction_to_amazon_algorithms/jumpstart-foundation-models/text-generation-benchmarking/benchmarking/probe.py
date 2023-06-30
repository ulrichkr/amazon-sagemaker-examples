import copy
from enum import Enum
import math
import time
from sagemaker.predictor import Predictor

from benchmarking.load_test import run_load_test


MAX_ITERS = 25


class ProbeDimensions(str, Enum):
    """Parameters explored via probe."""

    INPUT_LENGTH = "input_length"
    MAX_NEW_TOKENS = "max_new_tokens"
    CONCURRENT_REQUESTS = "concurrent_requests"


def _llm_payload_with_fixed_new_tokens(input_length: int, max_new_tokens: int, temperature: int = 900):
    return {
        "inputs": "A " * input_length,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
    }


def probe(
    predictor: Predictor,
    dimension: ProbeDimensions,
    input_length: int = 500,
    max_new_tokens: int = 500,
    concurrent_requests: int = 1,
    scale_factor: float = 1.2,
) -> int:
    current_attempt = {
        ProbeDimensions.INPUT_LENGTH.value: input_length,
        ProbeDimensions.MAX_NEW_TOKENS.value: max_new_tokens,
        ProbeDimensions.CONCURRENT_REQUESTS.value: concurrent_requests,
    }
    last_success = {}
    print(f"Running probe on {dimension.value}, starting with {current_attempt} ...")
    for iter in range(MAX_ITERS):
        try:
            payload = _llm_payload_with_fixed_new_tokens(
                current_attempt[ProbeDimensions.INPUT_LENGTH], current_attempt[ProbeDimensions.MAX_NEW_TOKENS]
            )
            statistics = run_load_test(
                predictor,
                payload,
                current_attempt[ProbeDimensions.CONCURRENT_REQUESTS],
                current_attempt[ProbeDimensions.CONCURRENT_REQUESTS],
            )
            last_success = copy.copy(current_attempt)
            print(f" - Success with {last_success}, duration {statistics.duration_seconds():.2f}s, iteration {iter}")
            current_attempt[dimension] = math.ceil(current_attempt[dimension] * scale_factor)
        except Exception as e:
            print(f" - Failed with {current_attempt} with error {e}.")
            print(" - Waiting for 3 minutes to allow endpoint time to recover.")
            time.sleep(180.0)
            break
    return last_success


def run_probe(predictor: Predictor):
    return [
        probe(predictor, dimension=ProbeDimensions.MAX_NEW_TOKENS, input_length=100),
        probe(predictor, dimension=ProbeDimensions.MAX_NEW_TOKENS, input_length=1000),
        probe(predictor, dimension=ProbeDimensions.INPUT_LENGTH, max_new_tokens=100),
        probe(predictor, dimension=ProbeDimensions.INPUT_LENGTH, max_new_tokens=500),
        probe(predictor, dimension=ProbeDimensions.CONCURRENT_REQUESTS, max_new_tokens=100),
        probe(predictor, dimension=ProbeDimensions.CONCURRENT_REQUESTS, input_length=1000, max_new_tokens=100),
    ]
