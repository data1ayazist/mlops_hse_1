import yaml
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

# Откройте и загрузите YAML-файл
with open("configs/config.yml") as config_file:
    config = yaml.safe_load(config_file)

# Извлеките данные для подключения
MINIO_ENDPOINT = config["MINIO_ENDPOINT"]
ACCESS_KEY = config["MINIO_ACCESS_KEY"]
SECRET_KEY = config["MINIO_SECRET_KEY"]
BUCKET_NAME = config["MINIO_BUCKET_NAME"]
# Создайте сессию с использованием MinIO
session = boto3.session.Session()
s3 = session.client(
    "s3",
    endpoint_url=f"http://{MINIO_ENDPOINT}",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='us-east-1'
)


def create_bucket(bucket_name):
    s3.create_bucket(Bucket=bucket_name)
    print(f'Bucket "{bucket_name}" created.')



# Функция для создания папки в S3
def create_folder(bucket_name, folder_name):
    try:
        s3.put_object(Bucket=bucket_name, Key=(folder_name + "/"))
        print(f'Folder "{folder_name}" created in bucket "{bucket_name}".')
    except (NoCredentialsError, PartialCredentialsError):
        print("Credentials not available.")
    except Exception as e:
        print(f"Error creating folder: {e}")


if __name__ == "__main__":
    # Создайте бакет
    create_bucket(BUCKET_NAME)
    # Создайте папки "models" и "data"
    create_folder(BUCKET_NAME, "models")
    create_folder(BUCKET_NAME, "data")
