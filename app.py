from fastapi import FastAPI, UploadFile, File, Query, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
import pandas as pd
from data_models import ModelType
import yaml
import json
from io import StringIO
from typing import Annotated
import logging
from endpoints import app_train_model, app_retrain_model, app_predict, app_delete_model, get_memory_info

app = FastAPI()

# Настройка логгирования
log_file_path = "app.log"  # Укажите путь к файлу для сохранения логов
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),  # Сохранение логов в файл
        logging.StreamHandler(),  # Вывод логов в консоль
    ],
)
logger = logging.getLogger(__name__)

# Настройка базовой аутентификации
security = HTTPBasic()


# Функция для проверки аутентификации
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    logger.info("Попытка входа")
    with open("configs/config.yml", "r") as file:
        config = yaml.safe_load(file)
    correct_username = config["username"]
    correct_password = config["password"]
    if (
        credentials.username != correct_username
        or credentials.password != correct_password
    ):
        logger.info("Неверные креды")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверные учетные данные",
            headers={"WWW-Authenticate": "Basic"},
        )


@app.post("/train/")
async def train_model_endpoint(
    model_type: Annotated[
        str, Query(description="Тип модели, которую нужно создать")
    ] = "SVC",
    model_name: Annotated[
        str, Query(description="Имя модели, которую нужно создать")
    ] = "my_model",
    params: UploadFile = File(...),
    file: UploadFile = File(...),
    credentials: HTTPBasicCredentials = Depends(authenticate),
) -> JSONResponse:
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
        return JSONResponse(
            content={"error": "Такой тип модели не поддерживается, см. /model-types/"},
            status_code=400,
        )
    params_content = await params.read()
    # Преобразуем содержимое в словарь
    params_dict = json.loads(params_content)
    # Чтение данных из загруженного CSV файла
    contents = await file.read()
    data = pd.read_csv(StringIO(contents.decode("utf-8")))
    try:
        app_train_model(
            data=data,
            model_type=model_type,
            params_dict=params_dict,
            model_name=model_name,
            config_path="configs/config.yml",
        )
        logger.info("Модель успешно обучена и сохранена.")
        return JSONResponse(content={"message": "Модель успешно обучена и сохранена."})
    except Exception as e:
        logger.exception("Ошибка при обучении модели: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/model-types/")
async def get_model_types(
    credentials: HTTPBasicCredentials = Depends(authenticate),
) -> dict:
    """
    Получает список типов моделей.

    Returns:
        dict: Словарь с доступными типами моделей.
    """
    logger.exception("Запрошены типы моделей")
    return {"model_types": [{model_type: model_type.value} for model_type in ModelType]}


@app.get("/healthcheck")
async def healthcheck(
    credentials: HTTPBasicCredentials = Depends(authenticate),
) -> dict:
    """
    Проверяет состояние сервиса.

    Returns:
        dict: Словарь с информацией о состоянии сервиса.
    """
    logger.exception("Запрошен healthcheck")
    used, free = get_memory_info()
    return {"status": "healthy", "Занятая память ОЗУ": f"{used} байт", 
            "Свободная память ОЗУ": f"{free} байт"}


@app.post("/retrain/")
async def retrain_model_endpoint(
    model_name: Annotated[
        str, Query(description="Имя модели, которую нужно переобучить")
    ],
    file: UploadFile = File(...),
    credentials: HTTPBasicCredentials = Depends(authenticate),
) -> JSONResponse:
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
        new_data = pd.read_csv(StringIO(contents.decode("utf-8")))

        status_retrain = app_retrain_model(
            new_data, model_name, config_path="configs/config.yml"
        )

        if not status_retrain:
            logger.exception("Модель не найдена")
            return JSONResponse(
                content={"error": "Модель не найдена."}, status_code=404
            )
        else:
            logger.info("Модель успешно переобучена и сохранена.")
            return JSONResponse(
                content={"message": "Модель успешно переобучена и сохранена."}
            )
    except Exception as e:
        logger.exception("Ошибка при переобучении модели: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post("/predict/")
async def predict_model_endpoint(
    model_name: Annotated[
        str, Query(description="Имя модели, через которую нужно дать прогноз")
    ],
    file: UploadFile = File(...),
    credentials: HTTPBasicCredentials = Depends(authenticate),
) -> JSONResponse:
    """
    Выполняет предсказание с использованием указанной модели.

    Args:
        model_name (str): Имя модели, через которую нужно дать прогноз.
        file (UploadFile): CSV файл с данными для предсказания.

    Returns:
        JSONResponse: Ответ с предсказаниями или сообщением об ошибке.
    """
    # try:
        # Чтение данных из загруженного CSV файла
    contents = await file.read()
    inference_data = pd.read_csv(StringIO(contents.decode("utf-8")))
    status_predict, predictions = app_predict(
        inference_data, model_name, config_path="configs/config.yml"
    )

    # Проверка, существует ли модель
    if not status_predict:
        return JSONResponse(
            content={"error": "Модель не найдена."}, status_code=404
        )
    else:
        logger.info("Прогноз успешно получен")
        return JSONResponse(
            content={
                "predictions": predictions.tolist()  # Преобразуем массив предсказаний в список
            }
        )
    # except Exception as e:
    #     logger.info("Ошибка при инференсе: %s", str(e))
    #     return JSONResponse(content={"error": str(e)}, status_code=400)


@app.delete("/delete_model/")
async def delete_model_endpoint(
    model_name: Annotated[str, Query(description="Имя модели, которую нужно удалить")],
    credentials: HTTPBasicCredentials = Depends(authenticate),
) -> JSONResponse:
    """
    Удаляет указанную модель.

    Args:
        model_name (str): Имя модели, которую нужно удалить.

    Returns:
        JSONResponse: Ответ с сообщением о результате операции.
    """
    try:
        delete_status = app_delete_model(model_name, config_path="configs/config.yml")
        if not delete_status:
            return JSONResponse(
                content={"error": "Модель не найдена."}, status_code=404
            )
        else:
            logger.info("Модель успешно удалена")
            return JSONResponse(
                content={"message": "Модель успешно удалена."}, status_code=200
            )

    except Exception as e:
        logger.info("Ошибка при удалении: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)
