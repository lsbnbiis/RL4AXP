import os
import torch as T
import numpy as np
import pandas as pd

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tf_keras as keras

ENCODING_TABLE: dict[str, list[float]] = pd.read_csv(os.path.join("./amp_prediction", "pc6_table.csv"), index_col=0).apply(list, axis=1).to_dict()

MODEL = keras.models.load_model(os.path.join("./amp_prediction", "ai4amp_model.h5"), compile=False)

DEVICE = T.device("cuda:0") if T.cuda.is_available() else T.device("cpu")

def batch_encode_peps(peptides: list[str]) -> np.ndarray:

    pep_vectors: list[list[float]] = []

    for pep in peptides:
        pep: str = pep + "X" * (200 - len(pep))
        pep_vectors.append([ENCODING_TABLE.get(aa, [0.0] * 6) for aa in pep])

    return np.array(pep_vectors)

def get_amp_probs(peptides: list[str]) -> T.Tensor:

    if len(peptides) == 0:
        return T.empty(0, dtype=T.float32, device=DEVICE)

    amp_probs: np.ndarray = MODEL.predict(batch_encode_peps(peptides), verbose=0, batch_size=512)

    return T.tensor(amp_probs, dtype=T.float32, device=DEVICE).reshape(-1)

if __name__ == "__main__":

    import time

    for _ in range(3): _ = get_amp_probs(["FLTHILKGLFTAGKVIHGLIHRRRHG"] * 10)

    peptides = ["FLTHILKGLFTAGKVIHGLIHRRRHG"] * 500

    times = []
    for _ in range(20):

        t1 = time.time()
        probs = get_amp_probs(peptides)
        t2 = time.time()

        times.append(t2 - t1)

    print(f"Stable inference time for 500 sequences: {np.mean(times):.4f} ± {np.std(times):.4f} sec")
        
    print(get_amp_probs(["SIGTAVKKAVPIAKKVGKVAIPIAKAVLSVVGQLVG"]))