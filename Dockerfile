# Используем официальный образ Python
FROM python:3.11.4-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы lock и toml в контейнер
COPY pyproject.toml .
COPY poetry.lock .

# Устанавливаем Poetry
RUN pip install poetry

# Устанавливаем зависимости, включая Uvicorn
RUN poetry install --no-root --no-dev

# Устанавливаем Uvicorn отдельно, если он не установлен через Poetry
RUN poetry add uvicorn

# Копируем остальные файлы приложения
COPY . .
Создаём папку временного буфера
# Указываем команду для запуска приложения
CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

