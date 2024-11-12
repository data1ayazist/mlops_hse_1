import grpc
import model_service_pb2
import model_service_pb2_grpc
from google.protobuf.empty_pb2 import Empty  # Импортируем Empty

def run():
    # Устанавливаем соединение с gRPC-сервером
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        
        # Вызываем метод GetModelTypes
        response = stub.GetModelTypes(Empty())  # Используем Empty вместо model_service_pb2.Empty
        
        # Выводим ответ
        print("Доступные типы моделей:")
        for model_type in response.model_types:
            print(model_type.name)

if __name__ == '__main__':
    run()
