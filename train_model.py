import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


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
    trained_model = model(**params)
    trained_model.fit(X, y)

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