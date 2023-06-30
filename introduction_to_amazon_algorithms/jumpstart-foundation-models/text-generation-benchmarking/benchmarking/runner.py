from concurrent import futures
from typing import Any, Tuple, Union
from typing import Dict
from typing import List

from sagemaker.jumpstart.model import JumpStartModel
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.session import Session
from sagemaker.utils import name_from_base

from benchmarking.constants import MAX_CONCURRENT_INVOCATIONS_PER_MODEL
from benchmarking.constants import MAX_TOTAL_RETRY_TIME_SECONDS
from benchmarking.constants import NUM_INVOCATIONS
from benchmarking.constants import RETRY_WAIT_TIME_SECONDS
from benchmarking.constants import SM_SESSION
from benchmarking.load_test import run_benchmarking_load_tests
from benchmarking.load_test import logging_prefix


class Benchmarker:

    def __init__(
        self,
        payloads: Dict[str, Dict[str, Any]],
        max_concurrent_benchmarks: int,
        sagemaker_session: Session = SM_SESSION,
        num_invocations: int = NUM_INVOCATIONS,
        max_workers: int = MAX_CONCURRENT_INVOCATIONS_PER_MODEL,
        retry_wait_time: float = RETRY_WAIT_TIME_SECONDS,
        max_total_retry_time: float = MAX_TOTAL_RETRY_TIME_SECONDS,
        run_latency_load_test: str = True,
        run_throughput_load_test: str = True,
    ):
        self.payloads = payloads
        self.max_concurrent_benchmarks = max_concurrent_benchmarks
        self.sagemaker_session = sagemaker_session
        self.num_invocations = num_invocations
        self.max_workers = max_workers
        self.retry_wait_time = retry_wait_time
        self.max_total_retry_time = max_total_retry_time
        self.run_latency_load_test = run_latency_load_test
        self.run_throughput_load_test = run_throughput_load_test

    def run_single_predictor(self, model_id: str, predictor: Predictor, clean_up: bool = True) -> List[Dict[str, Any]]:
        metrics = []
        try:
            for payload_name, payload in self.payloads.items():
                metrics_payload = run_benchmarking_load_tests(
                    predictor=predictor,
                    payload=payload,
                    model_id=model_id,
                    payload_name=payload_name,
                    num_invocations=self.num_invocations,
                    max_workers=self.max_workers,
                    retry_wait_time=self.retry_wait_time,
                    max_total_retry_time=self.max_total_retry_time,
                    run_latency_load_test=self.run_latency_load_test,
                    run_throughput_load_test=self.run_throughput_load_test,
                )
                metrics.append(metrics_payload)
        finally:
            if clean_up is True:
                print(f"{logging_prefix(model_id)} Cleaning up resources ...")
                predictor.delete_model()
                predictor.delete_endpoint()
            else:
                print(f"{logging_prefix(model_id)} Skipping cleaning up resources ...")

        return metrics

    def run_single_model_id(self, model_id: str) -> List[Dict[str, Any]]:
        model = JumpStartModel(model_id=model_id, sagemaker_session=self.sagemaker_session)
        endpoint_name = name_from_base(f"jumpstart-bm-{model_id.replace('huggingface', 'hf')}")
        print(f"{logging_prefix(model_id)} Deploying endpoint {endpoint_name} ...")
        predictor = model.deploy(endpoint_name=endpoint_name)
        predictor.serializer = JSONSerializer()
        predictor.content_type = "application/json"
        return self.run_single_predictor(model_id, predictor)


    def run_multiple_model_ids(self, models: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        metrics = []
        errors = {}

        with futures.ThreadPoolExecutor(max_workers=self.max_concurrent_benchmarks) as executor:
            future_to_model_id = {
                executor.submit(self.run_single_model_id, model_id): model_id for model_id in models
            }
            for future in futures.as_completed(future_to_model_id):
                model_id = future_to_model_id[future]
                try:
                    metrics.extend(future.result())
                except Exception as e:
                    errors[model_id] = e
                    print(f"(Model {model_id}) Benchmarking failed: {e}")

        return metrics, errors
