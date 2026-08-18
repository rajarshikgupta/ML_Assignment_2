from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

OUT = Path("model")
OUT.mkdir(exist_ok=True)

digits = load_digits()
X = pd.DataFrame(digits.data, columns=[f"pixel_{i+1}" for i in range(64)])
y = pd.Series(digits.target, name="target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=42))
    ]),
    "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=20),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)

    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, prediction),
        "AUC": roc_auc_score(y_test, probability, multi_class="ovr", average="weighted"),
        "Precision": precision_score(y_test, prediction, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, prediction, average="weighted", zero_division=0),
        "F1": f1_score(y_test, prediction, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, prediction),
    })
    joblib.dump(model, OUT / f"{name}.joblib")

pd.DataFrame(rows).to_csv("model_metrics.csv", index=False)

test_data = X_test.copy()
test_data["target"] = y_test.to_numpy()
test_data.to_csv("test_data.csv", index=False)

print(pd.DataFrame(rows).round(4))
