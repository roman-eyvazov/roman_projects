Работающее веб-приложение на Streamlit, которое позволяет рассчитать цену квартиры по заданным параметрам. Для прогнозирования используется предобученный XGBoost.

Ввиду того, что в настоящее время (май 2026 г.) регистрация на Streamlit без VPN не работает, приложение не развернуто, но работает локально. При запуске необходимо разархивировать XGBoost_model.rar в папку с файлом Apartment_price_analysis_Streamlit.py. Файл ColumnTransformer должен быть в той же папке.

Данные получены отсюда (актуальны на 2021 г.):

https://www.kaggle.com/datasets/mrdaniilak/russia-real-estate-2021

Блокнот на Kaggle с EDA и обучением модели:

https://www.kaggle.com/code/romaneyvazov/russia-real-estate-analysis/notebook
