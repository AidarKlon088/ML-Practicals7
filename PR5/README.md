# Практическая работа №5 — Логистическая регрессия

**Тема:** Бинарная классификация  
**Датасет:** Spotify Tracks Dataset  
**Библиотеки:** Pandas, NumPy, Matplotlib, Scikit-learn

## О проекте

Я обучил модель определять популярность музыкальных треков Spotify. Треки с популярностью от 50 относятся к классу 1, остальные — к классу 0.

## Основной код

| Код | Что делает |
|---|---|
| `train_test_split()` | Делит данные на train и test |
| `SimpleImputer()` | Заполняет пропуски |
| `StandardScaler()` | Масштабирует числа |
| `OneHotEncoder()` | Преобразует жанры в числовые признаки |
| `Pipeline()` | Объединяет обработку и модель |
| `LogisticRegression()` | Обучает классификатор |
| `model.predict()` | Предсказывает класс трека |
| `confusion_matrix()` | Показывает правильные и ошибочные предсказания |
| `accuracy_score()` | Считает общую точность |
| `precision_score()` | Проверяет точность положительных предсказаний |
| `recall_score()` | Показывает, сколько популярных треков найдено |
| `f1_score()` | Считает баланс Precision и Recall |

## Результат

Я обучил модель, построил матрицу ошибок и оценил её с помощью Accuracy, Precision, Recall и F1. Реальные значения метрик находятся в `notebook.ipynb`.

## Запуск

1. Установить зависимости: `pip install pandas numpy matplotlib scipy scikit-learn jupyter`
2. Открыть `notebook.ipynb` в VS Code.
3. Выбрать Python Kernel и нажать **Run All**.

Используется `PR1/dataset.csv`.