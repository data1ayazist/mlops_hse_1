import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def train_model(model_type, data: pd.DataFrame, params: dict = None):
    # Предполагаем, что последний столбец - это целевая переменная
    X = data.iloc[:, :-1]  # Все столбцы, кроме последнего
    y = data.iloc[:, -1]   # Последний столбец
    # Выбор модели
    if model_type == 'SVC':
        model = SVC
    elif model_type == 'RandomForest':
        model = RandomForestClassifier
    else:
        raise ValueError("Неподдерживаемый тип модели. Используйте 'SVC' или 'RandomForest'.")

    # Настройка гиперпараметров с помощью GridSearchCV
    print(params)
    trained_model = model(**params)
    trained_model.fit(X, y)

    return trained_model

def retrain_model(trained_model, new_data):
    X = new_data.iloc[:, :-1]  # Все столбцы, кроме последнего
    y = new_data.iloc[:, -1]   # Последний столбец
    trained_model.fit(X, y)

    return trained_model