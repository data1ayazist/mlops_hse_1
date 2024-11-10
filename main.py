from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
from typing import Dict, Any
from train_model import train_model, retrain_model
import joblib
import os
import yaml
from enum import Enum
import json
from io import StringIO
from pydantic import BaseModel

app = FastAPI()

class ModelType(str, Enum):
    SVC = "SVC"
    RandomForest = "RandomForest"

app = FastAPI()


@app.post("/train/")
async def train_model_endpoint(
    model_type: str,
    model_name: str,
    params: UploadFile = File(...),
    file: UploadFile = File(...)
):
    # Читаем содержимое файла params
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
        return JSONResponse(content={
            "message": "Модель успешно обучена и сохранена."
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    
@app.get("/model-types/")
async def get_model_types():
    return {"model_types": [{model_type: model_type.value} for model_type in ModelType]}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "healthy"}

@app.post("/retrain/")
async def retrain_model_endpoint(
    model_name: str,
    file: UploadFile = File(...)
):
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
            return JSONResponse(content={"error": "Модель не найдена."}, status_code=404)
        
        # Загрузка существующей модели
        trained_model = joblib.load(model_path)
        
        # Переобучение модели на новых данных
        updated_model = retrain_model(trained_model, new_data)  # Предполагается, что у вас есть функция retrain_model
        
        # Сохранение обновленной модели
        joblib.dump(updated_model, model_path)
        
        return JSONResponse(content={
            "message": "Модель успешно переобучена и сохранена."
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    
@app.post("/predict/")
async def predict_model_endpoint(
    model_name: str,
    file: UploadFile = File(...)
):
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
        
        return JSONResponse(content={
            "predictions": predictions.tolist()  # Преобразуем массив предсказаний в список
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    
@app.delete("/delete_model/")
async def delete_model_endpoint(model_name: str):
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
        return JSONResponse(content={"message": "Модель успешно удалена."}, status_code=200)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
