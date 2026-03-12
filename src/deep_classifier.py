"""
deep_classifier.py
------------------
Deep learning classification for hyperspectral data.

Methods
-------
autoencoder  : Spectral Autoencoder (MLP) → latent-space K-means.
               Fully unsupervised – no labels needed.
               Better feature extraction than raw-spectra K-means.

cnn          : 1D-CNN pixel-wise classifier.
               Supervised – requires a labelled pixels CSV.
               Learns spectral shape patterns; outperforms RF on most
               hyperspectral datasets once enough labels are available.

PyTorch is required:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    # or CUDA version:
    pip install torch --index-url https://download.pytorch.org/whl/cu121
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# ===================================================================
# Model definitions
# ===================================================================

def _get_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        raise ImportError(
            "PyTorch is required for deep learning methods.\n"
            "CPU-only install:  pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu"
        )


class SpectralAutoencoder:
    """MLP autoencoder that compresses pixel spectra to a latent space."""

    def __init__(self, n_bands: int, latent_dim: int = 16):
        torch, nn = _get_torch()

        class _AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(n_bands, 256), nn.BatchNorm1d(256), nn.ReLU(),
                    nn.Linear(256, 128),     nn.BatchNorm1d(128), nn.ReLU(),
                    nn.Linear(128, 64),      nn.BatchNorm1d(64),  nn.ReLU(),
                    nn.Linear(64, latent_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 64),  nn.BatchNorm1d(64),  nn.ReLU(),
                    nn.Linear(64, 128),          nn.BatchNorm1d(128), nn.ReLU(),
                    nn.Linear(128, 256),         nn.BatchNorm1d(256), nn.ReLU(),
                    nn.Linear(256, n_bands),
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

            def encode(self, x):
                return self.encoder(x)

        self._model_cls = _AE
        self.n_bands    = n_bands
        self.latent_dim = latent_dim
        self.model      = None
        self._torch     = torch
        self._nn        = nn

    def train(
        self,
        pixels: np.ndarray,
        epochs: int = 60,
        batch_size: int = 1024,
        lr: float = 1e-3,
    ) -> List[float]:
        """Train autoencoder on pixel spectra. Returns per-epoch loss."""
        torch = self._torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"  AE training on {device} | "
                    f"{len(pixels):,} pixels | {epochs} epochs")

        model = self._model_cls().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = self._nn.MSELoss()

        X = torch.FloatTensor(pixels).to(device)
        dataset = torch.utils.data.TensorDataset(X)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        history = []
        model.train()
        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            for (batch,) in loader:
                optim.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                optim.step()
                ep_loss += loss.item()
            avg = ep_loss / len(loader)
            history.append(avg)
            if ep % max(1, epochs // 8) == 0 or ep == epochs:
                logger.info(f"  AE [{ep:>3}/{epochs}] loss={avg:.5f}")

        self.model = model
        return history

    def encode(self, pixels: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Encode all pixels to latent vectors. Returns (N, latent_dim)."""
        torch = self._torch
        device = next(self.model.parameters()).device
        self.model.eval()
        parts = []
        with torch.no_grad():
            for i in range(0, len(pixels), batch_size):
                chunk = torch.FloatTensor(pixels[i:i + batch_size]).to(device)
                parts.append(self.model.encode(chunk).cpu().numpy())
        return np.concatenate(parts, axis=0)


class SpectralCNN:
    """1D-CNN classifier for pixel-wise spectral classification."""

    def __init__(self, n_bands: int, n_classes: int):
        torch, nn = _get_torch()

        # Compute feature-map size after two MaxPool1d(2) layers
        after_pool = n_bands // 2 // 2
        after_pool = max(after_pool, 1)

        class _CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    # (batch, 1, n_bands)
                    nn.Conv1d(1,   32, kernel_size=7, padding=3), nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(32,  64, kernel_size=5, padding=2), nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(8),   # fixed output length = 8
                )
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 8, 256), nn.ReLU(), nn.Dropout(0.4),
                    nn.Linear(256, 128),     nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(128, n_classes),
                )

            def forward(self, x):
                # x: (batch, n_bands) → (batch, 1, n_bands)
                return self.head(self.features(x.unsqueeze(1)))

        self._model_cls = _CNN
        self.n_bands    = n_bands
        self.n_classes  = n_classes
        self.model      = None
        self._torch     = torch
        self._nn        = nn

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 512,
        lr: float = 1e-3,
        patience: int = 15,
    ) -> Dict[str, List[float]]:
        """Train CNN. Returns history dict with 'train_loss', 'val_acc'."""
        torch = self._torch
        nn    = self._nn
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            f"  CNN training on {device} | "
            f"{len(X_train):,} train samples | "
            f"{self.n_classes} classes | {epochs} epochs"
        )

        model   = self._model_cls().to(device)
        optim   = torch.optim.Adam(model.parameters(), lr=lr)
        sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=5, factor=0.5, verbose=False
        )
        loss_fn = nn.CrossEntropyLoss()

        Xt = torch.FloatTensor(X_train)
        yt = torch.LongTensor(y_train)
        train_ds = torch.utils.data.TensorDataset(Xt, yt)
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )

        has_val = X_val is not None and y_val is not None
        history: Dict[str, List] = {"train_loss": [], "val_acc": []}
        best_val, best_ep, best_state = 0.0, 0, None

        for ep in range(1, epochs + 1):
            model.train()
            ep_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optim.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optim.step()
                ep_loss += loss.item()

            avg_loss = ep_loss / len(train_dl)
            history["train_loss"].append(avg_loss)

            # Validation
            val_acc = 0.0
            if has_val:
                model.eval()
                with torch.no_grad():
                    Xv = torch.FloatTensor(X_val).to(device)
                    yv = torch.LongTensor(y_val).to(device)
                    preds = model(Xv).argmax(1)
                    val_acc = (preds == yv).float().mean().item()
                history["val_acc"].append(val_acc)
                sched.step(1 - val_acc)

                if val_acc > best_val:
                    best_val  = val_acc
                    best_ep   = ep
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}

                # Early stopping
                if ep - best_ep >= patience:
                    logger.info(f"  CNN early stop at epoch {ep} "
                                f"(best val_acc={best_val:.3f} @ ep{best_ep})")
                    break
            else:
                sched.step(avg_loss)

            if ep % max(1, epochs // 8) == 0 or ep == epochs:
                msg = f"  CNN [{ep:>3}/{epochs}] loss={avg_loss:.4f}"
                if has_val:
                    msg += f"  val_acc={val_acc:.3f}"
                logger.info(msg)

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)
            logger.info(f"  Restored best weights from epoch {best_ep}")

        self.model = model
        return history

    def predict(self, pixels: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Return class-id predictions for all pixels."""
        torch = self._torch
        device = next(self.model.parameters()).device
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(pixels), batch_size):
                chunk = torch.FloatTensor(pixels[i:i + batch_size]).to(device)
                preds.append(self.model(chunk).argmax(1).cpu().numpy())
        return np.concatenate(preds, axis=0)

    def predict_proba(self, pixels: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Return softmax probability matrix (N, n_classes)."""
        torch = self._torch
        nn    = self._nn
        device = next(self.model.parameters()).device
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(pixels), batch_size):
                chunk = torch.FloatTensor(pixels[i:i + batch_size]).to(device)
                p = nn.functional.softmax(self.model(chunk), dim=1)
                probs.append(p.cpu().numpy())
        return np.concatenate(probs, axis=0)


# ===================================================================
# High-level interface
# ===================================================================

class DeepClassifier:
    """
    Orchestrates deep-learning classification workflows.
    Called from HyperspectralClassifier when method='autoencoder' or 'cnn'.
    """

    def __init__(self, config: dict):
        self.cfg = config.get("classification", {})

    # ---------------------------------------------------------- #
    # Autoencoder + K-means (unsupervised)
    # ---------------------------------------------------------- #

    def classify_autoencoder(
        self,
        data: np.ndarray,
        n_clusters: int,
    ) -> np.ndarray:
        """
        1. Train spectral autoencoder on a pixel sample.
        2. Encode ALL pixels to latent space.
        3. K-means on latent vectors.
        Returns class_map (H, W), 1-indexed.
        """
        from sklearn.cluster import KMeans

        ae_cfg     = self.cfg.get("autoencoder", {})
        H, W, B    = data.shape
        flat       = data.reshape(-1, B).astype(np.float32)

        # Random subsample for training
        max_px = int(ae_cfg.get("max_pixels", 100_000))
        if len(flat) > max_px:
            rng  = np.random.default_rng(42)
            idx  = rng.choice(len(flat), size=max_px, replace=False)
            train_px = flat[idx]
        else:
            train_px = flat

        # Train autoencoder
        ae = SpectralAutoencoder(
            n_bands=B,
            latent_dim=int(ae_cfg.get("latent_dim", 16)),
        )
        ae.train(
            train_px,
            epochs=int(ae_cfg.get("epochs", 60)),
            batch_size=int(ae_cfg.get("batch_size", 1024)),
            lr=float(ae_cfg.get("learning_rate", 1e-3)),
        )

        # Encode all pixels
        logger.info(f"  AE encoding {len(flat):,} pixels...")
        latent = ae.encode(flat)                      # (N, latent_dim)

        # K-means on latent space
        logger.info(f"  K-means ({n_clusters} clusters) on latent space...")
        km     = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(latent)               # 0-indexed

        class_map = (labels + 1).reshape(H, W)        # 1-indexed
        return class_map

    # ---------------------------------------------------------- #
    # 1D-CNN supervised classifier
    # ---------------------------------------------------------- #

    def classify_cnn(
        self,
        data: np.ndarray,
        n_classes: int,
        labels_csv: Optional[str],
    ) -> np.ndarray:
        """
        1. Load labelled pixels from CSV (row, col, class_id).
        2. Train 1D-CNN.
        3. Predict all pixels.
        Returns class_map (H, W).
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        if labels_csv is None:
            raise ValueError(
                "cnn method requires --labels CSV with columns: row, col, class_id"
            )

        cnn_cfg = self.cfg.get("cnn", {})
        H, W, B = data.shape
        flat    = data.reshape(-1, B).astype(np.float32)

        # Load labels
        df   = pd.read_csv(labels_csv, names=["row", "col", "class_id"])
        rows = df["row"].astype(int).values
        cols = df["col"].astype(int).values
        y_raw = df["class_id"].astype(int).values

        # Normalise class IDs to 0..n-1
        le = LabelEncoder()
        y  = le.fit_transform(y_raw).astype(np.int64)
        actual_n_classes = len(le.classes_)

        X = data[rows, cols, :].astype(np.float32)    # (N_labels, B)
        logger.info(
            f"  CNN labels: {len(X):,} pixels | "
            f"{actual_n_classes} classes | "
            f"class IDs: {le.classes_.tolist()}"
        )

        # Train / val split
        test_split = float(cnn_cfg.get("test_split", 0.2))
        if len(X) >= 20:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=test_split, stratify=y, random_state=42
            )
        else:
            X_tr, X_val, y_tr, y_val = X, None, y, None
            logger.warning("  Too few labels for validation split")

        # Train CNN
        cnn = SpectralCNN(n_bands=B, n_classes=actual_n_classes)
        cnn.train(
            X_tr, y_tr, X_val, y_val,
            epochs=int(cnn_cfg.get("epochs", 100)),
            batch_size=int(cnn_cfg.get("batch_size", 512)),
            lr=float(cnn_cfg.get("learning_rate", 1e-3)),
            patience=int(cnn_cfg.get("patience", 15)),
        )

        # Predict all pixels
        logger.info(f"  CNN predicting {len(flat):,} pixels...")
        pred_encoded = cnn.predict(flat)              # 0-indexed encoded labels
        pred_orig    = le.inverse_transform(pred_encoded)   # original class IDs
        class_map    = pred_orig.reshape(H, W)

        return class_map
