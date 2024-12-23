import boto3
import joblib
import os


def get_s3_client(config: dict):
    # Откройте и загрузите YAML-файл
    # Извлеките данные для подключения
    MINIO_ENDPOINT = config["MINIO_ENDPOINT"]
    ACCESS_KEY = config["MINIO_ACCESS_KEY"]
    SECRET_KEY = config["MINIO_SECRET_KEY"]
    session = boto3.session.Session()
    s3 = session.client(
        "s3",
        endpoint_url=f"http://{MINIO_ENDPOINT}",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    return s3


def save_model(trained_model, s3_client, config, model_name):
    bucket_name = config["MINIO_BUCKET_NAME"]
    models_dir = config["models_dir"]
    model_path = f"{models_dir}/{model_name}.joblib"
    temp_buffer_path = config["TEMP_BUFFER_PATH"]
    if not os.path.exists(temp_buffer_path):
        os.makedirs(temp_buffer_path)
    temp_model_path = os.path.join(temp_buffer_path, f"{model_name}.joblib")
    joblib.dump(trained_model, temp_model_path)
    s3_client.upload_file(temp_model_path, bucket_name, model_path)


def get_model(s3_client, config, model_name):
    bucket_name = config["MINIO_BUCKET_NAME"]
    temp_buffer_path = config["TEMP_BUFFER_PATH"]
    models_dir = config["models_dir"]
    model_path = f"{models_dir}/{model_name}.joblib"
    if not os.path.exists(temp_buffer_path):
        os.makedirs(temp_buffer_path)
    temp_model_path = os.path.join(temp_buffer_path, f"{model_name}.joblib")
    s3_client.download_file(bucket_name, model_path, temp_model_path)
    trained_model = joblib.load(temp_model_path)
    os.remove(temp_model_path)
    return trained_model
