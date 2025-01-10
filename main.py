from math import ceil

import telebot
import pandas as pd
import joblib
from pandas import Index

num_cols = Index(['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea','MiscVal', 'MoSold', 'YrSold', 'SalePrice'])

df = pd.read_csv("data.csv")
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

default_values = {}
for feature in numerical_features:
    default_values[feature] = df[feature].median()
for feature in categorical_features:
    default_values[feature] = df[feature].mode()[0]

# Загрузка предобученной модели и скейлера
model = joblib.load('xgb.joblib')
scaler = joblib.load('scaler.joblib')

# Создание бота
TOKEN = 'TOKEN'
bot = telebot.TeleBot(TOKEN)

# Диапазоны допустимых значений для признаков
feature_ranges = {
    'Качество постройки': (1, 10),     # OverallQual: от 1 до 10
    'Жилая площадь (м²)': (30, 500),  # GrLivArea в метрах (после перевода)
    'Количество машин в гараже': (0, 4)  # GarageCars: от 0 до 4
}

# Сообщение о диапазонах значений
validation_message = (
    "Введите данные в формате:\n"
    "Качество постройки, Жилая площадь (м²), Количество машин в гараже\n"
    "Пример: 7, 200, 2\n\n"
    f"Диапазоны значений:\n"
    f" - Качество постройки: {feature_ranges['Качество постройки'][0]}-{feature_ranges['Качество постройки'][1]}\n"
    f" - Жилая площадь: {feature_ranges['Жилая площадь (м²)'][0]}-{feature_ranges['Жилая площадь (м²)'][1]} м²\n"
    f" - Количество машин в гараже: {feature_ranges['Количество машин в гараже'][0]}-{feature_ranges['Количество машин в гараже'][1]}"
)

# Функция обработки старта
@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, validation_message)

# Функция обработки текста
@bot.message_handler(content_types=['text'])
def predict_price(message):
    if message.text[0] != "/":
        try:
            # Разделение и преобразование ввода
            user_input = message.text.split(',')
            if len(user_input) != 3:
                raise ValueError("Неверное количество данных! Убедитесь, что ввели 3 значения через запятую.")

            # Преобразование к числовому формату
            features = [int(value.strip()) for value in user_input]
            feature_names = ['Качество постройки', 'Жилая площадь (м²)', 'Количество машин в гараже']


            # Валидация диапазонов
            for name, value in zip(feature_names, features):
                min_val, max_val = feature_ranges[name]
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"Признак '{name}' имеет недопустимое значение ({value}). Диапазон: {min_val}-{max_val}.")

            features[1] = ceil(features[1] * 10.764)  # м² -> футы

            input_data = df.iloc[0:1].copy()
            for column in df.columns:
                if column in ['OverallQual', 'GrLivArea', 'GarageCars']:
                    input_data[column] = {
                        'OverallQual': features[0],
                        'GrLivArea': features[1],
                        'GarageCars': features[2]
                    }[column]
                else:
                    input_data[column] = default_values[column]

            input_data[num_cols] = scaler.transform(input_data[num_cols])

            input_data = input_data.drop("SalePrice", axis=1)
            prediction = model.predict(input_data)[0]

            predict_data = input_data.copy()
            predict_data["SalePrice"] = prediction
            predict_data[num_cols] = scaler.inverse_transform(predict_data[num_cols])
            bot.send_message(message.chat.id, f"Предполагаемая цена жилья: {ceil(predict_data["SalePrice"].iloc[0])} $")
        except ValueError as e:
            bot.send_message(message.chat.id, f"Ошибка ввода: {e}")
        except Exception as e:
            bot.send_message(message.chat.id, f"Ошибка обработки: {e}. Пожалуйста, проверьте формат ввода!")

bot.polling()
