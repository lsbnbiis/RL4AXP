"""
Trains the in-house MRSA-specific activity classifier.

Unlike the other predictors in this repo (AI4AMP, AI4ACP, AI4AVP,
LysisPeptica — all wrapped pretrained third-party models), no dedicated
anti-MRSA classifier with a ready-to-load pretrained model was available, so
this one is trained here directly.

Dataset source: Shoombuatong et al., "Advancing the Accuracy of Anti-MRSA
Peptide Prediction Through Integrating Multi-Source Protein Language
Models," Interdiscip Sci Comput Life Sci (2025).
https://github.com/Shoombuatong/pLM4MRSA (Dataset/TR_*.txt, TS_*.txt)

The published train/test split is kept intact: only TR_pos/TR_neg are used
for training (with a stratified slice held out for early stopping); TS_pos/
TS_neg are touched exactly once, at the end, to report an honest test score.
"""

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tf_keras as keras
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tf_keras import layers

_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_DIR, "data")
_PC6_PATH = os.path.join(_DIR, "6-pc")
_MODEL_PATH = os.path.join(_DIR, "mrsa_model.h5")
PAD_LEN = 70


def _build_pc6_table() -> dict[str, list[float]]:
    df = pd.read_csv(_PC6_PATH, sep=" ", index_col=0)

    def _zscore(col: pd.Series) -> np.ndarray:
        return (col - col.mean()) / col.std(ddof=1)

    matrix = np.array([
        _zscore(df["H1"]), _zscore(df["V"]), _zscore(df["P1"]),
        _zscore(df["Pl"]), _zscore(df["PKa"]), _zscore(df["NCI"]),
    ])

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    table = {aa: matrix[:, i].tolist() for i, aa in enumerate(amino_acids)}
    table["X"] = [0.0] * 6
    return table


def _read_fasta(path: str) -> list[str]:
    lines = [l.strip() for l in open(path) if l.strip()]
    return [l for l in lines if not l.startswith(">")]


def batch_encode_peps(peptides: list[str], table: dict[str, list[float]], length: int = PAD_LEN) -> np.ndarray:
    vectors = []
    for pep in peptides:
        pep = pep.ljust(length, "X")[:length]
        vectors.append([table.get(aa, [0.0] * 6) for aa in pep])
    return np.array(vectors, dtype=np.float32)


def build_model(input_len: int = PAD_LEN) -> keras.Model:
    inputs = keras.Input(shape=(input_len, 6))

    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(8, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)


def main() -> None:
    table = _build_pc6_table()

    tr_pos = _read_fasta(os.path.join(_DATA_DIR, "TR_pos.txt"))
    tr_neg = _read_fasta(os.path.join(_DATA_DIR, "TR_neg.txt"))
    ts_pos = _read_fasta(os.path.join(_DATA_DIR, "TS_pos.txt"))
    ts_neg = _read_fasta(os.path.join(_DATA_DIR, "TS_neg.txt"))

    X_tr_all = batch_encode_peps(tr_pos + tr_neg, table)
    y_tr_all = np.array([1] * len(tr_pos) + [0] * len(tr_neg), dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_all, y_tr_all, test_size=0.15, stratify=y_tr_all, random_state=3407,
    )

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    class_weight = {0: 1.0, 1: float(n_neg / n_pos)}

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)],
        verbose=2,
    )

    X_test = batch_encode_peps(ts_pos + ts_neg, table)
    y_test = np.array([1] * len(ts_pos) + [0] * len(ts_neg), dtype=np.float32)
    y_pred_prob = model.predict(X_test, verbose=0).squeeze()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print(f"\n=== Held-out test set (n={len(y_test)}, {int(y_test.sum())} positive) ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(y_test, y_pred_prob):.4f}")
    print(f"F1       : {f1_score(y_test, y_pred):.4f}")

    model.save(_MODEL_PATH)
    print(f"\nSaved model to {_MODEL_PATH}")


if __name__ == "__main__":
    main()
