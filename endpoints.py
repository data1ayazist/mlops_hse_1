import pandas as pd
from train_model import train_model, retrain_model
import joblib
import os
import yaml
from typing import Tuple, Dict, Any
import numpy as np


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
    model_path = os.path.join(config["models_dir"], f"{model_name}.joblib")
    joblib.dump(trained_model, model_path)


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

    model_path = os.path.join(config["models_dir"], f"{model_name}.joblib")

    # Проверка, существует ли модель
    if not os.path.exists(model_path):
        return False

    # Загрузка существующей модели
    trained_model = joblib.load(model_path)

    # Переобучение модели на новых данных
    updated_model = retrain_model(trained_model, new_data)

    # Сохранение обновленной модели
    joblib.dump(updated_model, model_path)

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

    model_path = os.path.join(config["models_dir"], f"{model_name}.joblib")

    # Проверка, существует ли модель
    if not os.path.exists(model_path):
        return False, None

    # Загрузка существующей модели
    trained_model = joblib.load(model_path)

    # Выполнение предсказания
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

    model_path = os.path.join(config["models_dir"], f"{model_name}.joblib")

    # Проверка, существует ли модель
    if not os.path.exists(model_path):
        return False

    # Удаление модели
    os.remove(model_path)
    return True