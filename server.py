import grpc
from concurrent import futures
from data_models import ModelType
from train_model import train_model, retrain_model
import model_service_pb2
import model_service_pb2_grpc
import pandas as pd
import joblib
import os
import yaml
import json
import logging
from io import StringIO
from endpoints import app_train_model, app_retrain_model, app_predict, app_delete_model

# Настройка логгирования
log_file_path = "app_grpc.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ModelService(model_service_pb2_grpc.ModelServiceServicer):

    def TrainModel(self, request, context):
        logger.info(
            "Запрос на обучение модели: тип=%s, имя=%s",
            request.model_type,
            request.model_name,
        )

        # Проверка типа модели
        if request.model_type not in [model_type.value for model_type in ModelType]:
            logger.error("Не поддерживаемый тип модели: %s", request.model_type)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Такой тип модели не поддерживается, см. /model-types/")
            return model_service_pb2.TrainModelResponse()

        # Чтение параметров
        params_dict = json.loads(request.params_json)

        # Чтение данных из загруженного CSV файла
        data = pd.read_csv(StringIO(request.file_content.decode("utf-8")))

        try:
            app_train_model(
                data=data,
                model_type=request.model_type,
                params_dict=params_dict,
                model_name=request.model_name,
                config_path="configs/config.yml",
            )
            logger.info("Модель успешно обучена и сохранена.")
            return model_service_pb2.TrainModelResponse(
                message="Модель успешно обучена и сохранена."
            )
        except Exception as e:
            logger.exception("Ошибка при обучении модели: %s", str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_service_pb2.TrainModelResponse()

    def RetrainModel(self, request, context):
        logger.info("Запрос на переобучение модели: имя=%s", request.model_name)

        # Чтение данных из загруженного CSV файла
        new_data = pd.read_csv(StringIO(request.file_content.decode("utf-8")))

        status_retrain = app_retrain_model(
            new_data, request.model_name, config_path="configs/config.yml"
        )
        # Проверка, существует ли модель
        if not status_retrain:
            logger.error("Модель не найдена: %s", request.model_name)
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Модель не найдена.")
            return model_service_pb2.RetrainModelResponse()
        else:
            logger.info("Модель успешно переобучена и сохранена.")
            return model_service_pb2.RetrainModelResponse(
                message="Модель успешно переобучена и сохранена."
            )

    def Predict(self, request, context):
        logger.info("Запрос на предсказание с моделью: имя=%s", request.model_name)

        # Чтение данных из загруженного CSV файла
        inference_data = pd.read_csv(StringIO(request.file_content.decode("utf-8")))
        status_predict, predictions = app_predict(
            inference_data, request.model_name, config_path="configs/config.yml"
        )

        # Проверка, существует ли модель
        if not status_predict:
            logger.error("Модель не найдена: %s", request.model_name)
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Модель не найдена.")
            return model_service_pb2.PredictResponse()
        else:
            logger.info("Прогноз успешно получен")
            return model_service_pb2.PredictResponse(
                predictions=predictions.tolist()
            )  # Преобразуем массив предсказаний в список

    def DeleteModel(self, request, context):
        logger.info("Запрос на удаление модели: имя=%s", request.model_name)
        delete_status = app_delete_model(request.model_name, config_path="configs/config.yml")
        if not delete_status:
            logger.error("Модель не найдена: %s", request.model_name)
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Модель не найдена.")
            return model_service_pb2.DeleteModelResponse()
        else:
            logger.info("Модель успешно удалена.")
            return model_service_pb2.DeleteModelResponse(message="Модель успешно удалена.")

    def GetModelTypes(self):
        logger.info("Запрошены типы моделей")
        model_types = [
            model_service_pb2.ModelType(name=model_type.value)
            for model_type in ModelType
        ]
        return model_service_pb2.ModelTypesResponse(model_types=model_types)

    def HealthCheck(self):
        logger.info("Запрошен healthcheck")
        return model_service_pb2.Empty()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(ModelService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    logger.info("gRPC сервер запущен на порту 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
