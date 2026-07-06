import os
import joblib
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from river import tree

MODEL_DIR = "models"
DATA_DIR = "data"

N_TREES = 5
RECORDS_PER_TREE = 1000
SAMPLE_SIZE = 30000

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def row_to_dict(row):
    return {f"f{i}": float(v) for i, v in enumerate(row)}


def load_kdd(sample_size=SAMPLE_SIZE):
    data = fetch_kddcup99(percent10=True, as_frame=True)
    df = data.frame.sample(sample_size, random_state=42).reset_index(drop=True)

    y = df["labels"].astype(str)
    y = (y != "b'normal.'").astype(int)

    X = df.drop(columns=["labels"])

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].apply(
                lambda v: v.decode("utf-8") if isinstance(v, bytes) else str(v)
            )

    return X, y


def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.drop(columns=cat_cols).columns.tolist()

    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])


def main():
    X_raw, y_raw = load_kdd()

    train_size = N_TREES * RECORDS_PER_TREE

    X_train_raw = X_raw.iloc[:train_size]
    y_train = y_raw.iloc[:train_size].values

    X_stream_raw = X_raw.iloc[train_size:].copy()
    y_stream = y_raw.iloc[train_size:].values

    preprocessor = build_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw).toarray()

    joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.pkl")

    for tree_id in range(N_TREES):
        model = tree.HoeffdingTreeClassifier()

        start = tree_id * RECORDS_PER_TREE
        end = start + RECORDS_PER_TREE

        print(f"Training Tree {tree_id}: records {start} to {end - 1}")

        for x, y in zip(X_train[start:end], y_train[start:end]):
            model.learn_one(row_to_dict(x), int(y))

        joblib.dump(model, f"{MODEL_DIR}/tree_{tree_id}.pkl")

    # Keep label in stream file for local/offline evaluation only.
    # The Spark drift detector does NOT use this label for drift detection.
    X_stream_raw["label"] = y_stream
    X_stream_raw.to_csv(f"{DATA_DIR}/kdd_stream.csv", index=False)

    # Also create a production-like copy with no label.
    X_stream_raw.drop(columns=["label"]).to_csv(
        f"{DATA_DIR}/kdd_stream_unlabeled.csv", index=False
    )

    print("Training complete.")
    print(f"Saved models in: {MODEL_DIR}/")
    print(f"Saved labeled stream for evaluation: {DATA_DIR}/kdd_stream.csv")
    print(f"Saved unlabeled stream for production-like test: {DATA_DIR}/kdd_stream_unlabeled.csv")


if __name__ == "__main__":
    main()