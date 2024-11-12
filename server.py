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

# Настройка логгирования
log_file_path = "app_grpc.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelService(model_service_pb2_grpc.ModelServiceServicer):

    def TrainModel(self, request, context):
        logger.info("Запрос на обучение модели: тип=%s, имя=%s", request.model_type, request.model_name)
        
        # Проверка типа модели
        if request.model_type not in [model_type.value for model_type in ModelType]:
            logger.error("Не поддерживаемый тип модели: %s", request.model_type)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Такой тип модели не поддерживается, см. /model-types/")
            return model_service_pb2.TrainModelResponse()

        # Чтение параметров
        params_dict = json.loads(request.params_json)
        
        # Чтение данных из загруженного CSV файла
        data = pd.read_csv(StringIO(request.file_content.decode('utf-8')))
        
        try:
            trained_model = train_model(request.model_type, data, params_dict)
            with open("configs/config.yml", "r") as file:
                config = yaml.safe_load(file)
            model_path = os.path.join(config['models_dir'], f'{request.model_name}.joblib')
            joblib.dump(trained_model, model_path)
            logger.info("Модель успешно обучена и сохранена: %s", model_path)
            return model_service_pb2.TrainModelResponse(message="Модель успешно обучена и сохранена.")
        except Exception as e:
            logger.exception("Ошибка при обучении модели: %s", str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_service_pb2.TrainModelResponse()

    def RetrainModel(self, request, context):
        logger.info("Запрос на переобучение модели: имя=%s", request.model_name)
        
        # Чтение данных из загруженного CSV файла
        new_data = pd.read_csv(StringIO(request.file_content.decode('utf-8')))
        
        # Загрузка конфигурации
        with open("configs/config.yml", "r") as file:
            config = yaml.safe_load(file)
        
        model_path = os.path.join(config['models_dir'], f'{request.model_name}.joblib')
        # Проверка, существует ли модель
        if not os.path.exists(model_path):
            logger.error("Модель не найдена: %s", request.model_name)
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Модель не найдена.")
            return model_service_pb2.RetrainModelResponse()

        # Загрузка существующей модели
        trained_model = joblib.load(model_path)

        # Переобучение модели на новых данных
        updated_model = retrain_model(trained_model, new_data)  # Предполагается, что у вас есть функция retrain_model

        # Сохранение обновленной модели
        joblib.dump(updated_model, model_path)
        logger.info("Модель успешно переобучена и сохранена: %s", model_path)
        return model_service_pb2.RetrainModelResponse(message="Модель успешно переобучена и сохранена.")

    def Predict(self, request, context):
        logger.info("Запрос на предсказание с моделью: имя=%s", request.model_name)

        # Чтение данных из загруженного CSV файла
        inference_data = pd.read_csv(StringIO(request.file_content.decode('utf-8')))

        # Загрузка конфигурации
        with open("configs/config.yml", "r") as file:
            config = yaml.safe_load(file)

        model_path = os.path.join(config['models_dir'], f'{request.model_name}.joblib')

        # Проверка, существует ли модель
        if not os.path.exists(model_path):
            logger.error("Модель не найдена: %s", request.model_name)
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Модель не найдена.")
            return model_service_pb2.PredictResponse()

        # Загрузка существующей модели
        trained_model = joblib.load(model_path)

        # Выполнение предсказания
        predictions = trained_model.predict(inference_data)  # Предполагается, что inference_data имеет правильный формат
        logger.info("Прогноз успешно получен")
        return model_service_pb2.PredictResponse(predictions=predictions.tolist())  # Преобразуем массив предсказаний в список

    def DeleteModel(self, request, context):
        logger.info("Запрос на удаление модели: имя=%s", request.model_name)

        # Загрузка конфигурации
        with open("configs/config.yml", "r") as file:
            config = yaml.safe_load(file)

        model_path = os.path.join(config['models_dir'], f'{request.model_name}.joblib')

        # Проверка, существует ли модель
        if not os.path.exists(model_path):
            logger.error("Модель не найдена: %s", request.model_name)
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Модель не найдена.")
            return model_service_pb2.DeleteModelResponse()

        # Удаление модели
        os.remove(model_path)
        logger.info("Модель успешно удалена: %s", model_path)
        return model_service_pb2.DeleteModelResponse(message="Модель успешно удалена.")

    def GetModelTypes(self, request, context):
        logger.info("Запрошены типы моделей")
        model_types = [model_service_pb2.ModelType(name=model_type.value) for model_type in ModelType]
        return model_service_pb2.ModelTypesResponse(model_types=model_types)

    def HealthCheck(self, request, context):
        logger.info("Запрошен healthcheck")
        return model_service_pb2.Empty()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(ModelService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("gRPC сервер запущен на порту 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
