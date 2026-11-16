import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError

BUCKET = "cse-cic-ids2018"
PREFIX = "Processed Traffic Data for ML Algorithms/"
# I am not sure which region hosts the bucket, so the script tries several
REGIONS = ["ca-central-1", "us-east-1", "us-west-2", "eu-west-1"]


def list_files():
    last_error = None
    for region in REGIONS:
        # UNSIGNED = no AWS account or password needed, the data is public
        s3 = boto3.client("s3", region_name=region, config=Config(signature_version=UNSIGNED))
        try:
            files, token = [], None
            while True:
                kwargs = {"Bucket": BUCKET, "Prefix": PREFIX}
                if token:
                    kwargs["ContinuationToken"] = token
                response = s3.list_objects_v2(**kwargs)
                files += [(o["Key"], o["Size"]) for o in response.get("Contents", [])]
                if not response.get("IsTruncated"):
                    break
                token = response["NextContinuationToken"]
            return region, files
        except (ClientError, EndpointConnectionError) as error:
            print(f"Region {region} did not work: {type(error).__name__}")
            last_error = error
    raise last_error


region, files = list_files()
files = sorted(f for f in files if not f[0].endswith("/"))

print(f"\nConnected using region: {region}")
print(f"Files found under '{PREFIX}': {len(files)}\n")
print(f"{'Size (MB)':>10}  File")
print("-" * 70)
for key, size in files:
    print(f"{size / 1_000_000:>10,.1f}  {key.replace(PREFIX, '')}")
print("-" * 70)
print(f"{sum(s for _, s in files) / 1_000_000_000:>9.2f} GB in total")