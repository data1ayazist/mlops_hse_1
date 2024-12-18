# mlops_hse_1

# Модельный сервис

Этот проект представляет собой сервис для обучения, переобучения и предсказания моделей машинного обучения с использованием FastAPI, Gradio и gRPC.

## Структура проекта

- **main.py**: FastAPI-сервис, который предоставляет RESTful API для работы с моделями.
- **gradio_app.py**: Приложение Gradio для интерактивного взаимодействия с моделями.
- **server.py**: gRPC-сервис для работы с моделями через gRPC.
- **configs/**: Папка с конфигурационными файлами, включая секреты и настройки.
- **models/**: Хранилище для сохраненных моделей машинного обучения.
- **tests/**: Папка для тестов, включая примеры файлов параметров модели и данных для обучения/инференса.

## Полезные команды

1. Запуск FastAPI-сервиса:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
2. Запуск Gradio-приложения:
   ```bash
   python gradio_app.py
3. Запуск gRPC-сервиса:
   ```bash
   python server.py
4. Запуск тестов для gRPC-сервиса:
   ```bash
   python test_grpc_server.py

# Установка и запуск приложения в Docker

Этот проект использует Docker для создания изолированного окружения и запуска приложения.

## Требования

- Установленный Docker на вашем компьютере. [Инструкции по установке Docker](https://docs.docker.com/get-docker/).

## Сборка Docker-образа

1. Скачайте или клонируйте репозиторий с проектом на ваш компьютер.
2. Перейдите в директорию проекта, где находится `Dockerfile`.
3. Выполните следующую команду для сборки Docker-образа:

   ```bash
   docker build -t myapp .
4. Запустите контейнер

   ```bash
   docker run -p 8000:8000 myapp


## Распределение ролей в команде
Валетдинов Аяз -  FastAPI. Полина Щукина - gradio+аутентификация. Сивков Александр - poetry+gRPC
