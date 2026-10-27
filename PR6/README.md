# Практическая работа №6 — Классификация текстов

**Тема:** Naive Bayes, CountVectorizer и TF-IDF  
**Язык:** Python  
**Библиотеки:** Pandas, Scikit-learn, Matplotlib

## О проекте

Я использовал [SMS Spam Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), чтобы обучить модель отличать обычные SMS от спама.

## Основной код

| Код | Что делает |
|---|---|
| `pd.read_csv()` | Загружает SMS |
| `train_test_split()` | Делит данные на train и test |
| `CountVectorizer()` | Подсчитывает слова |
| `TfidfVectorizer()` | Вычисляет веса слов |
| `MultinomialNB()` | Обучает классификатор |
| `Pipeline()` | Соединяет векторизацию и модель |
| `predict()` | Предсказывает класс SMS |
| `accuracy_score()` | Проверяет общую точность |
| `f1_score()` | Оценивает качество определения спама |
| `confusion_matrix()` | Показывает ошибки классификации |

## Что я сделал

Обучил две модели на одинаковых данных, сравнил Accuracy и F1, построил графики и посмотрел ошибочные предсказания.

## Запуск

1. Установить библиотеки: `pip install pandas scikit-learn matplotlib jupyter`
2. Открыть `notebook.ipynb` в VS Code.
3. Выбрать Python Kernel и нажать **Run All**.

Файл `spam.csv` находится в папке `PR6`.

## Результат

Сравнил CountVectorizer и TF-IDF с наивным байесовским классификатором. Метрики и результаты проверки находятся в ноутбуке.