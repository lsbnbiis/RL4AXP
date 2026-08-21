import os
import joblib
import torch as T
import numpy as np
import pandas as pd

from gensim.models import Doc2Vec
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

_PC6_PATH = os.path.join("./afp_prediction", "data", "6-pc")
_PC6_NN = os.path.join("./afp_prediction", "ensemble_model", "pc6", "pc6_best_weights.h5")
_PC6_SVM = os.path.join("./afp_prediction", "ensemble_model", "pc6", "pc6_features_svm.pkl")
_PC6_RF = os.path.join("./afp_prediction", "ensemble_model", "pc6", "pc6_features_forest.pkl")

_D2V_NN = os.path.join("./afp_prediction", "ensemble_model", "doc2vec", "doc2vec_best_weights.h5")
_D2V_SVM = os.path.join("./afp_prediction", "ensemble_model", "doc2vec", "doc2vec_features_svm.pkl")
_D2V_RF = os.path.join("./afp_prediction", "ensemble_model", "doc2vec", "doc2vec_features_forest.pkl")
_D2V_MDL = os.path.join("./afp_prediction", "Doc2Vec_model", "AFP_doc2vec.model")

_BERT_MDL = os.path.join("./afp_prediction", "ensemble_model", "bert", "ensemble_prot_bert_bfd_epoch1_1e-06.pt")
_BERT_NAME = "Rostlab/prot_bert_bfd"

_ENSEMBLE = os.path.join("./afp_prediction", "ensemble_model", "ensemble_best_weights.h5")

_PC6_THRES = 0.5191
_D2V_THRES = 0.3754
_BERT_THRES = 0.99
_PAD_LEN = 50

DEVICE = T.device("cuda:0") if T.cuda.is_available() else T.device("cpu")

def _build_pc6_table() -> dict[str, list[float]]:

    df = pd.read_csv(_PC6_PATH, sep=" ", index_col=0)

    def _zscore(col: pd.Series) -> np.ndarray:
        return (col - col.mean()) / col.std(ddof=1)
    
    matrix = np.array([_zscore(df[c]) for c in ("H1", "V", "P1", "Pl", "PKa", "NCI")])
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    table = {aa: matrix[:, i].tolist() for i, aa in enumerate(amino_acids)}
    table["X"] = [0.0] * 6

    return table

ENCODING_TABLE = _build_pc6_table()

PC6_NN = keras.models.load_model(_PC6_NN, compile=False)
PC6_SVM = joblib.load(_PC6_SVM)
PC6_RF = joblib.load(_PC6_RF)

D2V_NN = keras.models.load_model(_D2V_NN, compile=False)
D2V_SVM = joblib.load(_D2V_SVM)
D2V_RF = joblib.load(_D2V_RF)
D2V_MDL = Doc2Vec.load(_D2V_MDL)

_BERT_AVAILABLE = os.path.isfile(_BERT_MDL)
if _BERT_AVAILABLE:
    BERT_TOKENIZER = BertTokenizer.from_pretrained(_BERT_NAME, do_lower_case=False)
    BERT_MODEL = T.load(_BERT_MDL, map_location=DEVICE, weights_only=False)
    BERT_MODEL.config.output_attentions = False
    BERT_MODEL.config.output_hidden_states = False
    BERT_MODEL.eval()
else:
    import warnings
    warnings.warn(
        f"AFP BERT model not found at {_BERT_MDL}. "
        "BERT component will be replaced with a neutral (0.5) prediction.",
        RuntimeWarning, stacklevel=1
    )
    BERT_TOKENIZER = None
    BERT_MODEL = None

ENSEMBLE_MODEL = keras.models.load_model(_ENSEMBLE, compile=False)

def _pc6_encode(peptides: list[str], length: int = _PAD_LEN) -> np.ndarray:

    vecs = []
    for pep in peptides:
        pep = pep.ljust(length, "X")[:length]
        vecs.append([ENCODING_TABLE.get(aa, [0.0] * 6) for aa in pep])

    return np.array(vecs, dtype=np.float32)

def _doc2vec_encode(peptides: list[str], k: int = 3) -> np.ndarray:

    def _to_kmers(seq: str) -> list[str]:
        return [seq[i:i+k] for i in range(len(seq) - k + 1)]
    
    vecs = [D2V_MDL.infer_vector(_to_kmers(pep)) for pep in peptides]

    return np.array(vecs, dtype=np.float32)

def _bert_encode(peptides: list[str], max_len: int = _PAD_LEN) -> tuple[T.Tensor, T.Tensor]:

    input_ids, masks = [], []
    
    for pep in peptides:
        spaced = " ".join(list(pep[:max_len - 2]))

        enc = BERT_TOKENIZER.encode_plus(
            spaced,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids.append(enc["input_ids"][0])
        masks.append(enc["attention_mask"][0])

    return T.stack(input_ids), T.stack(masks)

def _run_pc6(peptides: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    data_3d = _pc6_encode(peptides)
    data_2d = data_3d.reshape(len(peptides), -1)
    nn_score = PC6_NN.predict(data_3d, verbose=0, batch_size=512).reshape(-1)
    svm_bin = PC6_SVM.predict(data_2d).astype(float)
    rf_bin = PC6_RF.predict(data_2d).astype(float)

    return nn_score, svm_bin, rf_bin

def _run_doc2vec(peptides: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    data = _doc2vec_encode(peptides)

    nn_score = D2V_NN.predict(data, verbose=0, batch_size=512).reshape(-1)
    svm_bin = D2V_SVM.predict(data).astype(float)
    rf_bin = D2V_RF.predict(data).astype(float)

    return nn_score, svm_bin, rf_bin

def _run_bert(peptides: list[str]) -> np.ndarray:

    input_ids, masks = _bert_encode(peptides)
    dataset = TensorDataset(input_ids, masks)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=1)

    logits_list = []
    with T.no_grad():
        for batch in dataloader:
            b_ids, b_mask = (t.to(DEVICE) for t in batch)
            outputs = BERT_MODEL(b_ids, token_type_ids=None, attention_mask=b_mask)
            logits_list.append(outputs[0].detach().cpu().numpy())

    logits = np.concatenate(logits_list, axis=0)
    probs  = T.nn.Softmax(dim=1)(T.tensor(logits)).numpy()

    return (probs[:, 1] > _BERT_THRES).astype(float)

def get_afp_probs(peptides: list[str]) -> T.Tensor:

    pc6_nn, pc6_svm, pc6_rf = _run_pc6(peptides)
    d2v_nn, d2v_svm, d2v_rf = _run_doc2vec(peptides)

    if _BERT_AVAILABLE:
        bert_bin = _run_bert(peptides)
    else:
        # BERT model unavailable — use neutral 0.5 as a soft placeholder
        bert_bin = np.full(len(peptides), 0.5, dtype=np.float64)

    pc6_nn_bin = (pc6_nn >= _PC6_THRES).astype(float)
    d2v_nn_bin = (d2v_nn >= _D2V_THRES).astype(float)

    ensemble_input = np.stack(
        [pc6_rf, pc6_svm, pc6_nn_bin,
         d2v_rf, d2v_svm, d2v_nn_bin,
         bert_bin],
        axis=1,
    ).astype(np.float32)

    scores = ENSEMBLE_MODEL.predict(ensemble_input, verbose=0, batch_size=512).reshape(-1)

    return T.tensor(scores, dtype=T.float32, device=DEVICE)

if __name__ == "__main__":

    import time

    for _ in range(3): get_afp_probs(["FLTHILKGLFTAGKVIHGLIHRRRHG"] * 10)

    peptides = ["FLTHILKGLFTAGKVIHGLIHRRRHG"] * 50

    times = []
    for _ in range(5):

        t1 = time.time()
        get_afp_probs(peptides)
        t2 = time.time()

        times.append(t2 - t1)

    print(f"Stable inference time for 50 sequences: {np.mean(times):.4f} ± {np.std(times):.4f} sec")

    print(f"{get_afp_probs(['GVIKAAKKVVKVLKNLF']).item():.8f}")
    print(f"{get_afp_probs(['LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES']).item():.8f}")
    print(f"{get_afp_probs(['RFRPPILRPPIRPPFRPPFRPPVRPPIRPPFRPPFRPPIGPFP']).item():.8f}")
    print(f"{get_afp_probs(['RLYRKVYGRLYRKVYGRLYRKVYGRLYRKVYGKKK']).item():.8f}")
    print(f"{get_afp_probs(['YVSCLFRGARCRVYSGRSCCFGYYCRRDFPGSIFGTCSRRNF']).item():.8f}")

    print(f"{get_afp_probs(['GGETGGEGKGMWFGPRL']).item():.8f}")
    print(f"{get_afp_probs(['SGHTKIKVAVDT']).item():.8f}")
    print(f"{get_afp_probs(['FGETSGETKGMWFGPRL']).item():.8f}")
    print(f"{get_afp_probs(['QQGEGGPYGGLSPLRFS']).item():.8f}")
    print(f"{get_afp_probs(['AFLTLTPGSHVDSYVEA']).item():.8f}")