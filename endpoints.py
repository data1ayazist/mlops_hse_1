import pandas as pd
from train_model import train_model, retrain_model
import os
import yaml
from typing import Tuple, Dict, Any
import numpy as np

import resource
from s3_client import get_s3_client, save_model, get_model
import os
from botocore.exceptions import ClientError


def app_train_model(
    data: pd.DataFrame,
    model_type: str,
    params_dict: Dict[str, Any],
    model_name: str,
    config_path: str,
) -> None:
    """
    Обучает модель и сохраняет ее на диск.

    :param data: Данные для обучения модели.
    :param model_type: Тип модели для обучения.
    :param params_dict: Словарь с параметрами для обучения модели.
    :param model_name: Имя для сохраненной модели.
    :param config_path: Путь к конфигурационному файлу.
    """
    trained_model = train_model(model_type, data, params_dict)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    s3_client = get_s3_client(config)
    save_model(trained_model, s3_client, config, model_name)


def app_retrain_model(
    new_data: pd.DataFrame, model_name: str, config_path: str
) -> bool:
    """
    Переобучает существующую модель на новых данных.

    :param new_data: Новые данные для переобучения модели.
    :param model_name: Имя модели, которую нужно переобучить.
    :param config_path: Путь к конфигурационному файлу.
    :return: True, если переобучение прошло успешно, иначе False.
    """
    # Загрузка конфигурации
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    s3_client = get_s3_client(config)
    bucket_name = config["MINIO_BUCKET_NAME"]
    models_dir = config["models_dir"]
    model_path = f"{models_dir}/{model_name}.joblib"
    try:
        s3_client.head_object(Bucket=bucket_name, Key=model_path)
    except ClientError:
        return False
    trained_model = get_model(s3_client, config, model_name)
    # Переобучение модели на новых данных
    updated_model = retrain_model(trained_model, new_data)
    # Сохранение обновленной модели
    save_model(updated_model, s3_client, config, model_name)
    return True


def app_predict(
    inference_data: pd.DataFrame, model_name: str, config_path: str
) -> Tuple[bool, np.ndarray]:
    """
    Выполняет предсказание с использованием обученной модели.

    :param inference_data: Данные для предсказания.
    :param model_name: Имя модели для предсказания.
    :param config_path: Путь к конфигурационному файлу.
    :return: Кортеж, где первый элемент - булево значение, указывающее на успех операции,
             а второй элемент - массив предсказаний или None, если модель не найдена.
    """
    # Загрузка конфигурации
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    s3_client = get_s3_client(config)
    bucket_name = config["MINIO_BUCKET_NAME"]
    models_dir = config["models_dir"]
    model_path = f"{models_dir}/{model_name}.joblib"
    try:
        s3_client.head_object(Bucket=bucket_name, Key=model_path)
    except ClientError:
        return False, None
    trained_model = get_model(s3_client, config, model_name)
    predictions = trained_model.predict(inference_data)

    return True, predictions


def app_delete_model(model_name: str, config_path: str) -> bool:
    """
    Удаляет модель с диска.

    :param model_name: Имя модели, которую нужно удалить.
    :param config_path: Путь к конфигурационному файлу.
    :return: True, если модель была успешно удалена, иначе False.
    """
    # Загрузка конфигурации
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    s3_client = get_s3_client(config)
    bucket_name = config["MINIO_BUCKET_NAME"]
    models_dir = config["models_dir"]
    model_path = f"{models_dir}/{model_name}.joblib"
    try:
        s3_client.head_object(Bucket=bucket_name, Key=model_path)
    except ClientError:
        return False
    s3_client.delete_object(Bucket=bucket_name, Key=model_path)
    return True


def get_memory_info() -> Tuple[int, int]:
    """
    Возвращает количество занятой и свободной оперативной памяти в байтах.

    :return: Кортеж, где первый элемент - количество занятой памяти,
             а второй элемент - количество свободной памяти.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    used_memory = usage.ru_maxrss * 1024  # ru_maxrss возвращает значение в килобайтах
    free_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf(
        "SC_AVPHYS_PAGES"
    )  # Свободная память
    return used_memory, free_memory
