import boto3
from botocore.config import Config
from sagemaker.session import Session


MAX_CONCURRENT_INVOCATIONS_PER_MODEL = 30
RETRY_WAIT_TIME_SECONDS = 30.0
MAX_TOTAL_RETRY_TIME_SECONDS = 120.0
NUM_INVOCATIONS = 10
SM_SESSION = Session(
    sagemaker_client=boto3.client(
        "sagemaker",
        config=Config(connect_timeout=5, read_timeout=60, retries={"max_attempts": 20}),
    )
)
