import pytest
import pandas as pd
import yaml
from unittest.mock import patch, MagicMock

# Импортируем вашу функцию
from endpoints import app_train_model

@pytest.fixture
def mock_get_s3_client():
    """Фикстура для мокирования функции get_s3_client."""
    with patch('s3_client.get_s3_client') as mock:
        yield mock

@pytest.fixture
def mock_save_model():
    """Фикстура для мокирования функции save_model."""
    with patch('s3_client.save_model') as mock:
        yield mock

@pytest.fixture
def mock_train_model():
    """Фикстура для мокирования функции train_model."""
    with patch('train_model.train_model') as mock:
        mock.return_value = MagicMock()  # Возвращаем мокированный объект модели
        yield mock

@pytest.fixture
def mock_config_file(tmp_path):
    """Фикстура для создания временного конфигурационного файла."""
    config_data = {
        'models_dir' : 'models',
        'username': "dungeon_master",
        'password': "Van Darkholme",
        'MINIO_ENDPOINT': 'localhost:9000',
        'MINIO_ACCESS_KEY': 'Gipsy_Kings',
        'MINIO_SECRET_KEY': 'Gipsy_Kings',
        'MINIO_BUCKET_NAME': 'mlops',
        'TEMP_BUFFER_PATH': 'temp_buffer'
    }
    config_path = f"{tmp_path}/config.yaml"
    with open(config_path, "w") as file:
        yaml.dump(config_data, file)
    return config_path

def test_app_train_model(mock_get_s3_client, mock_save_model, mock_train_model, mock_config_file):
    # Подготовка данных для теста
    data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    model_type = 'SVC'
    params_dict = {'C': 0.1}
    model_name = 'test_model'
    config_path = str(mock_config_file)

    # Вызов тестируемой функции
    app_train_model(data, model_type, params_dict, model_name, config_path)

    # Проверка, что функции были вызваны
    mock_train_model.assert_called_once_with(model_type, data, params_dict)
    mock_get_s3_client.assert_called_once()
    mock_save_model.assert_called_once_with(mock_train_model.return_value, mock_get_s3_client.return_value, yaml.safe_load(open(config_path, "r")), model_name)
