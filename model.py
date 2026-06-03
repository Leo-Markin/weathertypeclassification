import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pickle import dump, load

DATA_PATH = "weather_classification_data.csv"
MODEL_PATH = "weather_model.mw"

# Категориальные признаки и их возможные значения (для one-hot кодирования)
CAT_COLS = ["Cloud Cover", "Season", "Location"]
NUM_COLS = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)",
            "Atmospheric Pressure", "UV Index", "Visibility (km)"]
TARGET = "Weather Type"


def open_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def clean_outliers(df):
    df = df.copy()
    df["Humidity"] = df["Humidity"].clip(upper=100)
    df["Precipitation (%)"] = df["Precipitation (%)"].clip(upper=100)
    df["Atmospheric Pressure"] = df["Atmospheric Pressure"].clip(lower=870, upper=1085)
    df['Temperature'] = df['Temperature'].clip(upper=60)
    return df


def preprocess_data(df, test=True):
    df = clean_outliers(df)

    if test:
        y = df[TARGET]
        X = df.drop(TARGET, axis=1)
    else:
        X = df

    X = pd.get_dummies(X, columns=CAT_COLS, drop_first=True)
    X = X.astype(float)

    if test:
        return X, y
    return X


def fit_and_save_model(path=MODEL_PATH):
    df = open_data()
    X, y = preprocess_data(df, test=True)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, train_size=0.75, random_state=42, stratify=y_enc)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=4, random_state=42)
    model.fit(X_train_sc, y_train)

    test_pred = model.predict(X_test_sc)
    accuracy = accuracy_score(y_test, test_pred)
    print(f"Accuracy модели на тесте: {accuracy:.4f}")
    bundle = {
        "model": model,
        "scaler": scaler,
        "label_encoder": le,
        "feature_columns": feature_columns,
        "accuracy": accuracy,
    }
    with open(path, "wb") as file:
        dump(bundle, file)
    print(f"Модель сохранена в {path}")


def load_model(path=MODEL_PATH):
    with open(path, "rb") as file:
        bundle = load(file)
    return bundle


def predict(user_df, path=MODEL_PATH):
    bundle = load_model(path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    le = bundle["label_encoder"]
    feature_columns = bundle["feature_columns"]

    X = preprocess_data(user_df, test=False)
    X = X.reindex(columns=feature_columns, fill_value=0.0)

    X_sc = scaler.transform(X)

    pred_idx = model.predict(X_sc)[0]
    proba = model.predict_proba(X_sc)[0]

    prediction = le.inverse_transform([pred_idx])[0]
    proba_df = pd.DataFrame(
        {"Тип погоды": le.classes_, "Вероятность": proba}
    ).sort_values("Вероятность", ascending=False).reset_index(drop=True)

    return prediction, proba_df


if __name__ == "__main__":
    fit_and_save_model()
