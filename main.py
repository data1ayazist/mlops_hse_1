from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import pandas as pd
from data_models import ModelType
from train_model import train_model, retrain_model
import joblib
import os
import yaml
import json
from io import StringIO
from typing import  Annotated
import logging

app = FastAPI()

# Настройка логгирования
log_file_path = "app.log"  # Укажите путь к файлу для сохранения логов
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),  # Сохранение логов в файл
        logging.StreamHandler()  # Вывод логов в консоль
    ]
)
logger = logging.getLogger(__name__)

@app.post("/train/")
async def train_model_endpoint(
    model_type: Annotated[str, Query(description='Тип модели, которую нужно создать')] = 'SVC',
    model_name: Annotated[str, Query(description='Имя модели, которую нужно создать')] = 'my_model', 
    params: UploadFile = File(...),
    file: UploadFile = File(...)
)-> JSONResponse:
    """
    Обучает модель на основе предоставленных данных и параметров.

    - **model_type**: Тип модели, которую нужно создать (например, "SVC").
    - **model_name**: Имя модели, под которым она будет сохранена.
    - **params**: Файл с параметрами для обучения модели в формате JSON.
    - **file**: Файл с данными для обучения модели в формате CSV.

    Обучает модель и сохраняет её в соотвествующей директории, возвращает JSON-ответ с сообщением об успешном обучении модели или ошибкой.

    - **Успешный ответ**: 
        - Код: 200
        - Содержимое: {"message": "Модель успешно обучена и сохранена."}
    
    - **Ошибка**: 
        - Код: 400
        - Содержимое: {"error": "Описание ошибки"}
    """
    # Читаем содержимое файла params
    logger.info("Запрос на обучение модели: тип=%s, имя=%s", model_type, model_name)
    if not model_type in [model_type.value for model_type in ModelType]:
        logger.error("Не поддерживаемый тип модели: %s", model_type)
        return JSONResponse(content={"error":"Такой тип модели не поддерживается, см. /model-types/"}, status_code=400)
    params_content = await params.read()
    params_dict = json.loads(params_content)  # Преобразуем содержимое в словарь
    # Чтение данных из загруженного CSV файла
    contents = await file.read()
    data = pd.read_csv(StringIO(contents.decode('utf-8')))
    try:
        trained_model = train_model(model_type, data, params_dict)
        with open("configs/config.yml", "r") as file:
            config = yaml.safe_load(file)
        model_path = os.path.join(config['models_dir'], f'{model_name}.joblib')
        joblib.dump(trained_model, model_path)
        logger.info("Модель успешно обучена и сохранена: %s", model_path)
        return JSONResponse(content={
            "message": "Модель успешно обучена и сохранена."
        })
    except Exception as e:
        logger.exception("Ошибка при обучении модели: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)
    
@app.get("/model-types/")
async def get_model_types()-> dict:
    """
    Получает список типов моделей.

    Returns:
        dict: Словарь с доступными типами моделей.
    """
    logger.exception("Запрошены типы моделей")
    return {"model_types": [{model_type: model_type.value} for model_type in ModelType]}

@app.get("/healthcheck")
async def healthcheck()-> dict:
    """
    Проверяет состояние сервиса.

    Returns:
        dict: Словарь с информацией о состоянии сервиса.
    """
    logger.exception("Запрошен healthcheck")
    return {"status": "healthy"}

@app.post("/retrain/")
async def retrain_model_endpoint(
    model_name: Annotated[str, Query(description='Имя модели, которую нужно переобучить')],
    file: UploadFile = File(...)
)-> JSONResponse:
    """
    Переобучает модель на основе загруженных данных.

    Args:
        model_name (str): Имя модели, которую нужно переобучить.
        file (UploadFile): CSV файл с новыми данными для переобучения.

    Returns:
        JSONResponse: Ответ с сообщением о результате операции.
    """
    try:
        # Чтение данных из загруженного CSV файла
        contents = await file.read()
        new_data = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Загрузка конфигурации
        with open("configs/config.yml", "r") as file:
            config = yaml.safe_load(file)
        
        model_path = os.path.join(config['models_dir'], f'{model_name}.joblib')
        
        # Проверка, существует ли модель
        if not os.path.exists(model_path):
            logger.exception("Модель не найдена")
            return JSONResponse(content={"error": "Модель не найдена."}, status_code=404)
        
        # Загрузка существующей модели
        trained_model = joblib.load(model_path)
        
        # Переобучение модели на новых данных
        updated_model = retrain_model(trained_model, new_data)  # Предполагается, что у вас есть функция retrain_model
        
        # Сохранение обновленной модели
        joblib.dump(updated_model, model_path)
        logger.info("Модель успешно переобучена и сохранена: %s", model_path)
        return JSONResponse(content={
            "message": "Модель успешно переобучена и сохранена."
        })
    except Exception as e:
        logger.exception("Ошибка при переобучении модели: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)
    
@app.post("/predict/")
async def predict_model_endpoint(
    model_name: Annotated[str, Query(description='Имя модели, через которую нужно дать прогноз')],
    file: UploadFile = File(...)
)-> JSONResponse:
    """
    Выполняет предсказание с использованием указанной модели.

    Args:
        model_name (str): Имя модели, через которую нужно дать прогноз.
        file (UploadFile): CSV файл с данными для предсказания.

    Returns:
        JSONResponse: Ответ с предсказаниями или сообщением об ошибке.
    """
    try:
        # Чтение данных из загруженного CSV файла
        contents = await file.read()
        inference_data = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        # Загрузка конфигурации
        with open("configs/config.yml", "r") as file:
            config = yaml.safe_load(file)
        
        model_path = os.path.join(config['models_dir'], f'{model_name}.joblib')
        
        # Проверка, существует ли модель
        if not os.path.exists(model_path):
            return JSONResponse(content={"error": "Модель не найдена."}, status_code=404)
        
        # Загрузка существующей модели
        trained_model = joblib.load(model_path)
        
        # Выполнение предсказания
        predictions = trained_model.predict(inference_data)  # Предполагается, что new_data имеет правильный формат
        logger.info("Прогноз успешно получен")
        return JSONResponse(content={
            "predictions": predictions.tolist()  # Преобразуем массив предсказаний в список
        })
    except Exception as e:
        logger.info("Ошибка при инференсе: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)
    
@app.delete("/delete_model/")
async def delete_model_endpoint(model_name: Annotated[str, Query(description='Имя модели, которую нужно удалить')])-> JSONResponse:
    """
    Удаляет указанную модель.

    Args:
        model_name (str): Имя модели, которую нужно удалить.

    Returns:
        JSONResponse: Ответ с сообщением о результате операции.
    """
    try:
        # Загрузка конфигурации
        with open("configs/config.yml", "r") as file:
            config = yaml.safe_load(file)
        
        model_path = os.path.join(config['models_dir'], f'{model_name}.joblib')
        
        # Проверка, существует ли модель
        if not os.path.exists(model_path):
            return JSONResponse(content={"error": "Модель не найдена."}, status_code=404)
        
        # Удаление модели
        os.remove(model_path)
        logger.info("Модель успешно удалена")
        return JSONResponse(content={"message": "Модель успешно удалена."}, status_code=200)
    
    except Exception as e:
        logger.info("Ошибка при удалении: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)
    
