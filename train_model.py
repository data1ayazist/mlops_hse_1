import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import datetime
from sklearn.metrics import accuracy_score

def train_model(model_type, data: pd.DataFrame, params: dict = None):
    """
    Обучает модель на основе предоставленных данных и параметров.

    Args:
        model_type (str): Тип модели, которую нужно создать. Поддерживаемые типы: 'SVC', 'RandomForest'.
        data (pd.DataFrame): Данные для обучения, где последний столбец является целевой переменной.
        params (dict, optional): Гиперпараметры для настройки модели. По умолчанию None.

    Raises:
        ValueError: Если указанный тип модели не поддерживается.

    Returns:
        model: Обученная модель.
    """
    # Предполагаем, что последний столбец - это целевая переменная
    X = data.iloc[:, :-1]  # Все столбцы, кроме последнего
    y = data.iloc[:, -1]   # Последний столбец
    # Выбор модели
    if model_type == 'SVC':
        model = SVC
    elif model_type == 'RandomForest':
        model = RandomForestClassifier
    else:
        raise ValueError("Неподдерживаемый тип модели. ")

    # Настройка гиперпараметров с помощью GridSearchCV
    print(params)
    mlflow.set_tracking_uri(uri="http://mlflow:5000")
    mlflow.set_experiment(f"{model_type}_{datetime.datetime.now()}")
    trained_model = model(**params)
    trained_model.fit(X, y)
    y_pred = trained_model.predict(X)
    accuracy= accuracy_score(y, y_pred)
        # Регистрируем метрики в MLflow
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)

        # Сохраняем модель в MLflow
        mlflow.sklearn.log_model(trained_model, "model")
        model_info = mlflow.sklearn.log_model(
        sk_model=trained_model,
        registered_model_name=f"{model_type}_{datetime.datetime.now()}",
        artifact_path = 'app_model'
    )

    return trained_model

def retrain_model(trained_model, new_data):
    """
    Переобучает существующую модель на новых данных.

    Args:
        trained_model: Обученная модель, которую нужно переобучить.
        new_data (pd.DataFrame): Новые данные для переобучения, где последний столбец является целевой переменной.

    Returns:
        model: Переобученная модель.
    """
    X = new_data.iloc[:, :-1]  # Все столбцы, кроме последнего
    y = new_data.iloc[:, -1]   # Последний столбец
    trained_model.fit(X, y)

    return trained_model