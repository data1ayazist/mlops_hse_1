import os
import joblib
import pandas as pd
import gradio as gr
from sklearn.metrics import accuracy_score, recall_score, precision_score
import yaml
# Путь к папке с моделями
MODEL_DIR = "path/to/your/models"

def get_model_names():
    """Возвращает список доступных моделей в папке."""
    with open("configs/config.yml", "r") as file:
        config = yaml.safe_load(file)
    model_files = [f for f in os.listdir(config['models_dir']) if f.endswith('.joblib')]
    return [f[:-7] for f in model_files]  # Убираем '.joblib' из названий

def load_model(model_name):
    """Загружает модель по имени."""
    with open("configs/config.yml", "r") as file:
        config = yaml.safe_load(file)
    model_path = os.path.join(config['models_dir'], f"{model_name}.joblib")
    return joblib.load(model_path)

def predict_and_evaluate(model_name, csv_file):
    """Применяет модель к данным и вычисляет метрики."""
    # Загружаем модель
    model = load_model(model_name)
    
    # Загружаем данные из CSV
    new_data = pd.read_csv(csv_file.name)
    
    # Разделяем данные на X и y
    X = new_data.iloc[:, :-1]
    y = new_data.iloc[:, -1]
    
    # Делаем прогноз
    y_pred = model.predict(X)
    
    # Вычисляем метрики
    accuracy = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred, average='weighted')
    precision = precision_score(y, y_pred, average='weighted')
    
    return accuracy, recall, precision

# Создаем интерфейс Gradio
iface = gr.Interface(
    fn=predict_and_evaluate,
    inputs=[
        gr.Dropdown(choices=get_model_names(), label="Выберите модель"),  # Используем функцию для получения моделей
        gr.File(label="Загрузите CSV файл")
    ],
    outputs=[
        gr.Number(label="Accuracy"),
        gr.Number(label="Recall"),
        gr.Number(label="Precision")
    ],
    title="Модель прогнозирования",
    description="Выберите модель и загрузите CSV файл для оценки."
)

# Запускаем приложение с указанием хоста и порта
iface.launch(server_name="0.0.0.0", server_port=7860)