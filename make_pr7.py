import nbformat

nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_markdown_cell("""# Практическая работа № 7. Случайный лес

**Цель:** сравнить дерево решений и случайный лес, оценить качество классификации и определить наиболее важные признаки.

Для классификации создаётся целевая переменная: высокий доход — доход не ниже 70-го процентиля. Столбец дохода не используется как входной признак, чтобы избежать утечки целевой информации."""),
    nbformat.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv("customers_dirty.csv")

print("Размер датасета:", df.shape)
display(df.head())
print("Столбцы:", df.columns.tolist())
print("Пропуски:")
display(df.isna().sum())"""),
    nbformat.v4.new_code_cell("""# Создаём целевую переменную на основе дохода
if "monthly_income" not in df.columns:
    raise ValueError("В датасете не найден столбец monthly_income. Проверь название столбца.")

df = df.dropna(subset=["monthly_income"]).copy()
threshold = df["monthly_income"].quantile(0.70)
df["high_income"] = (df["monthly_income"] >= threshold).astype(int)

# Доход исключаем из X, потому что он использован для создания целевой переменной
X = df.drop(columns=["monthly_income", "high_income"])
y = df["high_income"]

# Удаляем столбцы, в которых нет ни одного значения
X = X.dropna(axis=1, how="all")

# Определяем числовые и категориальные признаки
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []
if numeric_features:
    transformers.append(("num", numeric_pipeline, numeric_features))
if categorical_features:
    transformers.append(("cat", categorical_pipeline, categorical_features))

preprocessor = ColumnTransformer(transformers=transformers)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Порог высокого дохода:", round(threshold, 2))
print("Размер train:", X_train.shape)
print("Размер test:", X_test.shape)
print("Числовые признаки:", numeric_features)
print("Категориальные признаки:", categorical_features)"""),
    nbformat.v4.new_code_cell("""# Обучаем дерево решений
tree_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42, max_depth=5))
])

tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)

# Обучаем случайный лес на том же разбиении данных
forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_leaf=2,
        n_jobs=-1
    ))
])

forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)

print("Дерево решений:")
print(classification_report(y_test, tree_predictions, zero_division=0))
print("Случайный лес:")
print(classification_report(y_test, forest_predictions, zero_division=0))"""),
    nbformat.v4.new_code_cell("""# Сравниваем модели по одинаковым метрикам
results = pd.DataFrame([
    {
        "Модель": "Decision Tree",
        "Accuracy": accuracy_score(y_test, tree_predictions),
        "F1": f1_score(y_test, tree_predictions, zero_division=0)
    },
    {
        "Модель": "Random Forest (100 деревьев)",
        "Accuracy": accuracy_score(y_test, forest_predictions),
        "F1": f1_score(y_test, forest_predictions, zero_division=0)
    }
])

display(results.round(3))

print("В случайном лесу 100 деревьев. Это распространённый базовый выбор, позволяющий объединить предсказания многих деревьев и получить более стабильный результат.")"""),
    nbformat.v4.new_code_cell("""# Важность признаков случайного леса
trained_preprocessor = forest_model.named_steps["preprocessor"]
trained_forest = forest_model.named_steps["model"]

feature_names = trained_preprocessor.get_feature_names_out()
importances = trained_forest.feature_importances_

importance_df = pd.DataFrame({
    "Признак": feature_names,
    "Важность": importances
}).sort_values("Важность", ascending=False)

display(importance_df.head(10).round(4))

top_features = importance_df.head(2)
print("Два наиболее важных признака:")
for _, row in top_features.iterrows():
    print(f"- {row['Признак']}: {row['Важность']:.4f}")

plot_data = importance_df.head(10).sort_values("Важность")
plt.figure(figsize=(10, 6))
plt.barh(plot_data["Признак"], plot_data["Важность"])
plt.xlabel("Важность признака")
plt.ylabel("Признак")
plt.title("Топ-10 важных признаков случайного леса")
plt.tight_layout()
plt.show()"""),
    nbformat.v4.new_markdown_cell("""## Выводы

1. Дерево решений и случайный лес обучены на одном и том же train-наборе и оценены на одном test-наборе.
2. Accuracy и F1 позволяют сравнить качество моделей. Лучшую модель определяем по полученным значениям, а не заранее.
3. Случайный лес объединяет предсказания 100 деревьев и часто работает стабильнее одиночного дерева.
4. Два самых важных признака определяются по таблице и графику выше.
5. Важность показывает, насколько признак использовался моделью для предсказаний, но не доказывает причинно-следственную связь.""")
]

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
}

nbformat.write(nb, "PR7/notebook.ipynb")
print("Ноутбук создан: PR7/notebook.ipynb")
