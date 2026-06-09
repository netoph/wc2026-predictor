"""
model/calibration.py
--------------------
Calibración de probabilidades por temperature scaling multiclase.

La idea: si el modelo es sobreconfiado (probas muy extremas) → T > 1 aplana.
         Si el modelo es subconfiado (probas muy cercanas a 1/3) → T < 1 agudiza.

Referencia: Guo et al. (2017) "On Calibration of Modern Neural Networks"
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


# ============================================================
# Utilidades
# ============================================================

def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax numéricamente estable por fila."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 1:
        logits = logits[np.newaxis, :]
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def log_loss_multiclass(p: np.ndarray, y: np.ndarray) -> float:
    """Log-loss multiclase (promedio por muestra)."""
    n = len(y)
    p_clipped = np.clip(p, 1e-12, 1.0)
    return -np.mean(np.log(p_clipped[np.arange(n), y]))


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Brier Score multiclase = MSE entre probas y one-hot."""
    n_classes = p.shape[1]
    one_hot = np.eye(n_classes)[y]
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


# ============================================================
# Temperature Scaler
# ============================================================

@dataclass
class TemperatureScaler:
    """
    Calibrador de temperatura a una sola escalar T.
    T > 1 → suaviza (modelo sobreconfiado)
    T < 1 → agudiza (modelo subconfiado)
    T = 1 → sin cambio
    """
    T: float = 1.0
    fitted: bool = False
    train_logloss_before: float = float("nan")
    train_logloss_after: float = float("nan")
    train_brier_before: float = float("nan")
    train_brier_after: float = float("nan")
    n_fit_samples: int = 0

    def fit(self, p_raw: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        """
        Ajusta T minimizando log-loss en (p_raw, y_true).

        p_raw  : (N, 3) probabilidades sin calibrar [p_home, p_draw, p_away]
        y_true : (N,)   etiquetas {0=local, 1=empate, 2=visitante}
        """
        p_raw = np.clip(np.asarray(p_raw, dtype=float), 1e-12, 1.0)
        y_true = np.asarray(y_true, dtype=int)

        if len(p_raw) < 10:
            logger.warning("Set de calibración muy pequeño (%d partidos). T=1.0 por defecto.", len(p_raw))
            self.T = 1.0
            self.fitted = False
            return self

        logp = np.log(p_raw)

        # Métricas antes de calibrar
        self.train_logloss_before = log_loss_multiclass(p_raw, y_true)
        self.train_brier_before   = brier_score(p_raw, y_true)

        def objective(logT: float) -> float:
            T = np.exp(logT)
            p_cal = softmax(logp / T)
            return log_loss_multiclass(p_cal, y_true)

        res = minimize_scalar(objective, bounds=(-2.0, 2.0), method="bounded")
        self.T = float(np.exp(res.x))

        p_cal = self.transform(p_raw)
        self.train_logloss_after = log_loss_multiclass(p_cal, y_true)
        self.train_brier_after   = brier_score(p_cal, y_true)
        self.fitted = True
        self.n_fit_samples = len(y_true)

        logger.info(
            "TemperatureScaler → T=%.4f | LogLoss %.4f→%.4f | Brier %.4f→%.4f | n=%d",
            self.T,
            self.train_logloss_before, self.train_logloss_after,
            self.train_brier_before, self.train_brier_after,
            self.n_fit_samples,
        )
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        """
        Aplica temperature scaling a probabilidades crudas.

        p_raw : (N, 3) o (3,) → devuelve misma forma calibrada
        """
        p_raw = np.clip(np.asarray(p_raw, dtype=float), 1e-12, 1.0)
        scalar_input = (p_raw.ndim == 1)
        if scalar_input:
            p_raw = p_raw[np.newaxis, :]

        logp = np.log(p_raw)
        p_cal = softmax(logp / self.T)

        return p_cal[0] if scalar_input else p_cal

    def describe(self) -> str:
        status = "✓" if self.fitted else "✗ (no ajustado)"
        return (
            f"TemperatureScaler {status}\n"
            f"  T               = {self.T:.4f}\n"
            f"  n_fit_samples   = {self.n_fit_samples}\n"
            f"  LogLoss (cal)   : {self.train_logloss_before:.4f} → {self.train_logloss_after:.4f}\n"
            f"  Brier  (cal)    : {self.train_brier_before:.4f}   → {self.train_brier_after:.4f}\n"
        )


# ============================================================
# Diagrama de calibración (reliability diagram)
# ============================================================

def reliability_diagram(
    p_cal: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    Genera un reliability diagram para las probabilidades calibradas.
    Solo usa la probabilidad del resultado más probable como eje X.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib no disponible, no se genera reliability diagram.")
        return

    from pathlib import Path

    p_max = p_cal.max(axis=1)
    correct = (p_cal.argmax(axis=1) == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mean_conf, mean_acc, counts = [], [], []

    for i in range(n_bins):
        mask = (p_max >= bins[i]) & (p_max < bins[i + 1])
        if mask.sum() > 0:
            mean_conf.append(p_max[mask].mean())
            mean_acc.append(correct[mask].mean())
            counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectamente calibrado")
    ax.plot(mean_conf, mean_acc, "o-", color="#2563EB", lw=2, label="Modelo")
    ax.set_xlabel("Confianza (probabilidad predicha)")
    ax.set_ylabel("Exactitud real")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Reliability diagram guardado en %s", save_path)
    else:
        plt.show()

    plt.close(fig)
