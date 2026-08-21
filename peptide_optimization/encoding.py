import os
import config
import torch as T
import torch.nn.functional as F

from peptide_optimization._utils import get_pepbert

class PeptideEncoder:

    def __init__(self) -> None:

        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa2idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        self.device = T.device("cuda:0") if T.cuda.is_available() else T.device("cpu")

        self.peptide_len = len(config.TARGET_PEPTIDE)

        self.sensing_matrix_path = os.path.join("./peptide_optimization", "logs", "phi.pt")
        self.sensing_matrix = self._get_sensing_matrix()

        self.pepbert_small_model, self.pepbert_small_tokenizer = get_pepbert(repo_id="dzjxzyd/PepBERT-small-UniParc")
        self.pepbert_large_model, self.pepbert_large_tokenizer = get_pepbert(repo_id="dzjxzyd/PepBERT-large-UniParc")

    def _get_sensing_matrix(self) -> T.Tensor:

        if os.path.exists(self.sensing_matrix_path): self.sensing_matrix = T.load(self.sensing_matrix_path)

        else:
            
            self.sensing_matrix = T.randn(32, self.peptide_len * 20) / (32 ** 0.5)
            T.save(self.sensing_matrix, self.sensing_matrix_path)
        
        return self.sensing_matrix
    
    def encode(self, peptides: list[str]) -> T.Tensor:
        
        if config.ENCODING_SCHEME == "One-Hot_Encoding": return self._one_hot_encoding(peptides)
        
        if config.ENCODING_SCHEME == "Compressive_Sensing": return self._compressive_sensing(peptides)
        
        if config.ENCODING_SCHEME == "PepBERT-small": return self._pepbert_small(peptides)
        
        if config.ENCODING_SCHEME == "PepBERT-large": return self._pepbert_large(peptides)
        
        raise ValueError(f"Unsupported 'config.ENCODING_SCHEME': {config.ENCODING_SCHEME}")

    def _one_hot_encoding(self, peptides: list[str]) -> T.Tensor:

        indices = T.tensor([[self.aa2idx[aa] for aa in pep] for pep in peptides], dtype=T.long)
        one_hot = F.one_hot(indices.to(self.device), num_classes=len(self.amino_acids))

        flatten = one_hot.view(len(peptides), -1)
        return flatten.to(T.float32)

    def _compressive_sensing(self, peptides: list[str]) -> T.Tensor:

        sparse_matrix = self._one_hot_encoding(peptides) # shape: (N, self.peptide_len * 20)

        return sparse_matrix @ self.sensing_matrix.to(self.device).T

    def _pepbert_small(self, peptides: list[str]) -> T.Tensor:

        sos_id = self.pepbert_small_tokenizer.token_to_id("[SOS]")
        eos_id = self.pepbert_small_tokenizer.token_to_id("[EOS]")

        ids = [[sos_id] + self.pepbert_small_tokenizer.encode(p).ids + [eos_id] for p in peptides]
        input_ids = T.tensor(ids, dtype=T.int64).to(self.device)

        encoder_mask = T.ones((len(peptides), 1, 1, input_ids.size(1)), dtype=T.int64).to(self.device)

        with T.no_grad(): embeds: T.Tensor = self.pepbert_small_model.encode(input_ids, encoder_mask)

        return embeds[:, 1:-1, :].mean(dim=1)

    def _pepbert_large(self, peptides: list[str]) -> T.Tensor:

        sos_id = self.pepbert_large_tokenizer.token_to_id("[SOS]")
        eos_id = self.pepbert_large_tokenizer.token_to_id("[EOS]")

        ids = [[sos_id] + self.pepbert_large_tokenizer.encode(p).ids + [eos_id] for p in peptides]
        input_ids = T.tensor(ids, dtype=T.int64).to(self.device)

        encoder_mask = T.ones((len(peptides), 1, 1, input_ids.size(1)), dtype=T.int64).to(self.device)

        with T.no_grad(): embeds: T.Tensor = self.pepbert_large_model.encode(input_ids, encoder_mask)

        return embeds[:, 1:-1, :].mean(dim=1)

    def get_position_embeddings(self, peptide: str) -> T.Tensor:
        """
        Return per-residue embeddings for a single peptide: shape (L, d_model).
        Uses PepBERT-large (or small if large is unavailable).
        Used by the SQA Quantum Refinement module to build the attention proxy.
        """
        if config.ENCODING_SCHEME in ("PepBERT-large", "PepBERT-small"):
            model = self.pepbert_large_model
            tokenizer = self.pepbert_large_tokenizer
        else:
            model = self.pepbert_small_model
            tokenizer = self.pepbert_small_tokenizer

        sos_id = tokenizer.token_to_id("[SOS]")
        eos_id = tokenizer.token_to_id("[EOS]")

        ids = [sos_id] + tokenizer.encode(peptide).ids + [eos_id]
        input_ids = T.tensor([ids], dtype=T.int64, device=self.device)
        encoder_mask = T.ones((1, 1, 1, input_ids.size(1)), dtype=T.int64, device=self.device)

        with T.no_grad():
            embeds = model.encode(input_ids, encoder_mask)   # (1, L+2, d_model)

        return embeds[0, 1:-1, :]   # (L, d_model)
