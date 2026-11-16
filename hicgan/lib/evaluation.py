import os
import re
import glob
import json
import logging
import numpy as np
import pandas as pd
from scipy import sparse, stats, ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


def compute_ssim_2d(img1: np.ndarray, img2: np.ndarray, data_range: float = None, win_size: int = 7) -> float:
    """Compute Structural Similarity Index (SSIM) between two 2D arrays using scipy.ndimage."""
    x = np.asarray(img1, dtype=np.float64)
    y = np.asarray(img2, dtype=np.float64)
    if x.shape != y.shape or x.size == 0:
        return float("nan")
    if data_range is None:
        data_range = float(max(x.max(), y.max()) - min(x.min(), y.min()))
    if data_range <= 1e-12:
        return 1.0 if np.allclose(x, y) else 0.0

    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    w = min(win_size, x.shape[0], x.shape[1])
    if w < 2:
        return float("nan")

    mu_x = ndimage.uniform_filter(x, size=w)
    mu_y = ndimage.uniform_filter(y, size=w)
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_xx = np.maximum(0.0, ndimage.uniform_filter(x * x, size=w) - mu_xx)
    sigma_yy = np.maximum(0.0, ndimage.uniform_filter(y * y, size=w) - mu_yy)
    sigma_xy = ndimage.uniform_filter(x * y, size=w) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    ssim_map = np.where(den > 0, num / den, 0.0)
    return float(np.mean(ssim_map))


def compute_psnr(gt: np.ndarray, pred: np.ndarray, data_range: float = None) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB."""
    x = np.asarray(gt, dtype=np.float64)
    y = np.asarray(pred, dtype=np.float64)
    mse = float(np.mean((x - y) ** 2))
    if mse <= 1e-15:
        return float("inf")
    if data_range is None:
        data_range = float(max(x.max(), y.max()) - min(x.min(), y.min()))
    if data_range <= 1e-12:
        data_range = 1.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute Cosine Similarity between two flattened arrays."""
    u = np.asarray(vec1, dtype=np.float64).ravel()
    v = np.asarray(vec2, dtype=np.float64).ravel()
    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))
    if norm_u <= 1e-12 or norm_v <= 1e-12:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def safe_corr(vec1: np.ndarray, vec2: np.ndarray):
    """Return (pearson_r, spearman_rho), matching pandas/scipy NaN behavior on constant arrays."""
    u = np.asarray(vec1, dtype=np.float64).ravel()
    v = np.asarray(vec2, dtype=np.float64).ravel()
    if u.size < 2 or np.ptp(u) == 0.0 or np.ptp(v) == 0.0:
        return float("nan"), float("nan")
    try:
        pr, _ = stats.pearsonr(u, v)
    except Exception:
        pr = float("nan")
    try:
        sr, _ = stats.spearmanr(u, v)
    except Exception:
        sr = float("nan")
    return float(pr), float(sr)


def _resolve_eval_window_size(run_or_pred_dir: str):
    """Resolve prediction windowSize (in bins) from predictions.pkl or hicgan_metadata.pkl if present."""
    import pickle
    if not run_or_pred_dir:
        return None
    d = os.path.abspath(run_or_pred_dir)
    parent = os.path.dirname(d)
    grandparent = os.path.dirname(parent)
    candidates = [
        os.path.join(d, "predictions.pkl"),
        os.path.join(parent, "predictions.pkl"),
        os.path.join(d, "hicgan_metadata.pkl"),
        os.path.join(parent, "hicgan_metadata.pkl"),
        os.path.join(grandparent, "hicgan_metadata.pkl"),
        os.path.join(parent, "train_run", "hicgan_metadata.pkl"),
        os.path.join(grandparent, "train_run", "hicgan_metadata.pkl"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            try:
                with open(cand, "rb") as fh:
                    obj = pickle.load(fh)
                if isinstance(obj, dict) and obj.get("windowSize"):
                    ws = int(obj["windowSize"])
                    if ws > 0:
                        return ws
            except Exception:
                pass
    return None


def compute_binary_classification_metrics(y_true_bin: np.ndarray, y_score: np.ndarray):
    """Compute AUROC, AUPRC, and best-threshold F1-score for binary labels and continuous scores."""
    y_true = np.asarray(y_true_bin, dtype=np.int32).ravel()
    scores = np.asarray(y_score, dtype=np.float64).ravel()
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan"), float("nan")

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
        auroc = float(roc_auc_score(y_true, scores))
        auprc = float(average_precision_score(y_true, scores))
        prec, rec, _ = precision_recall_curve(y_true, scores)
        denom = prec + rec
        f1_vals = np.divide(2.0 * prec * rec, denom, out=np.zeros_like(denom, dtype=np.float64), where=denom > 0)
        best_f1 = float(np.max(f1_vals)) if f1_vals.size > 0 else float("nan")
        return auroc, auprc, best_f1
    except Exception:
        return float("nan"), float("nan"), float("nan")


def parse_training_losses(train_out_dir: str) -> pd.DataFrame:
    """
    Parse epoch-by-epoch training and validation losses from `lossValues_*.npz`
    and/or `training.log` inside `train_out_dir`.
    Supports both Phase 1 cGAN (`train_gen_total=...`) and Phase 2 Diffusion (`train_total=...`).
    """
    rows = []
    log_path = os.path.join(train_out_dir, "training.log")
    if os.path.exists(log_path):
        cgan_pattern = re.compile(
            r"\[EPOCH-LOSS\]\s*epoch=(\d+)/(\d+)\s*\|\s*"
            r"train_gen_total=([0-9.eE+-]+),\s*train_pixel=([0-9.eE+-]+),\s*train_adv=([0-9.eE+-]+),\s*train_tv=([0-9.eE+-]+),\s*"
            r"train_disc_total=([0-9.eE+-]+),\s*train_disc_true=([0-9.eE+-]+),\s*train_disc_fake=([0-9.eE+-]+)\s*\|\s*"
            r"val_gen_total=([0-9.eE+-]+),\s*val_pixel=([0-9.eE+-]+),\s*val_adv=([0-9.eE+-]+),\s*val_tv=([0-9.eE+-]+),\s*val_disc_total=([0-9.eE+-]+)"
        )
        diff_pattern = re.compile(
            r"\[EPOCH-LOSS\]\s*epoch=(\d+)/(\d+)\s*\|\s*"
            r"train_total=([0-9.eE+-]+),\s*train_noise=([0-9.eE+-]+),\s*train_x0=([0-9.eE+-]+),\s*train_bg=([0-9.eE+-]+)\s*\|\s*"
            r"val_total=([0-9.eE+-]+),\s*val_noise=([0-9.eE+-]+),\s*val_x0=([0-9.eE+-]+),\s*val_bg=([0-9.eE+-]+)"
        )
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                md = diff_pattern.search(line)
                if md:
                    tr_tot = float(md.group(3))
                    va_tot = float(md.group(7))
                    rows.append({
                        "epoch": int(md.group(1)),
                        "model_type": "diffusion",
                        "train_total": tr_tot,
                        "val_total": va_tot,
                        "train_noise": float(md.group(4)),
                        "train_x0": float(md.group(5)),
                        "train_bg": float(md.group(6)),
                        "val_noise": float(md.group(8)),
                        "val_x0": float(md.group(9)),
                        "val_bg": float(md.group(10)),
                        "train_gen_total": tr_tot,
                        "val_gen_total": va_tot,
                        "train_pixel": np.nan,
                        "val_pixel": np.nan,
                        "train_adv": np.nan,
                        "val_adv": np.nan,
                        "train_tv": np.nan,
                        "val_tv": np.nan,
                        "train_disc_total": np.nan,
                        "val_disc_total": np.nan,
                        "train_disc_true": np.nan,
                        "train_disc_fake": np.nan,
                    })
                    continue
                m = cgan_pattern.search(line)
                if m:
                    rows.append({
                        "epoch": int(m.group(1)),
                        "model_type": "cgan",
                        "train_gen_total": float(m.group(3)),
                        "train_pixel": float(m.group(4)),
                        "train_adv": float(m.group(5)),
                        "train_tv": float(m.group(6)),
                        "train_disc_total": float(m.group(7)),
                        "train_disc_true": float(m.group(8)),
                        "train_disc_fake": float(m.group(9)),
                        "val_gen_total": float(m.group(10)),
                        "val_pixel": float(m.group(11)),
                        "val_adv": float(m.group(12)),
                        "val_tv": float(m.group(13)),
                        "val_disc_total": float(m.group(14)),
                    })
    if rows:
        df = pd.DataFrame(rows).drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch").reset_index(drop=True)
        return df

    npz_files = sorted(glob.glob(os.path.join(train_out_dir, "lossValues_*.npz")))
    if not npz_files:
        return pd.DataFrame()

    latest_npz = npz_files[-1]
    data = np.load(latest_npz)
    keys = set(data.keys())

    if "diffLossTrainEpochs" in keys and "diffLossValEpochs" in keys:
        n_ep = len(data["diffLossTrainEpochs"])
        tr_tot = np.asarray(data["diffLossTrainEpochs"], dtype=float)
        va_tot = np.asarray(data["diffLossValEpochs"], dtype=float)
        df = pd.DataFrame({
            "epoch": np.arange(1, n_ep + 1, dtype=int),
            "model_type": "diffusion",
            "train_total": tr_tot,
            "val_total": va_tot,
            "train_noise": np.asarray(data["diffNoiseLossTrainEpochs"], dtype=float) if "diffNoiseLossTrainEpochs" in keys else np.nan,
            "val_noise": np.asarray(data["diffNoiseLossValEpochs"], dtype=float) if "diffNoiseLossValEpochs" in keys else np.nan,
            "train_x0": np.asarray(data["diffX0LossTrainEpochs"], dtype=float) if "diffX0LossTrainEpochs" in keys else np.nan,
            "val_x0": np.asarray(data["diffX0LossValEpochs"], dtype=float) if "diffX0LossValEpochs" in keys else np.nan,
            "train_bg": np.asarray(data["diffBgLossTrainEpochs"], dtype=float) if "diffBgLossTrainEpochs" in keys else np.nan,
            "val_bg": np.asarray(data["diffBgLossValEpochs"], dtype=float) if "diffBgLossValEpochs" in keys else np.nan,
            "train_gen_total": tr_tot,
            "val_gen_total": va_tot,
            "train_pixel": np.nan,
            "val_pixel": np.nan,
            "train_adv": np.nan,
            "val_adv": np.nan,
            "train_tv": np.nan,
            "val_tv": np.nan,
            "train_disc_total": np.nan,
            "val_disc_total": np.nan,
            "train_disc_true": np.nan,
            "train_disc_fake": np.nan,
        })
        return df

    if "genLossTrainEpochs" in keys and "genLossValEpochs" in keys:
        n_ep = len(data["genLossTrainEpochs"])
        df = pd.DataFrame({
            "epoch": np.arange(1, n_ep + 1, dtype=int),
            "model_type": "cgan",
            "train_gen_total": np.asarray(data["genLossTrainEpochs"], dtype=float),
            "val_gen_total": np.asarray(data["genLossValEpochs"], dtype=float),
            "train_pixel": np.asarray(data["genPixelLossTrainEpochs"], dtype=float) if "genPixelLossTrainEpochs" in keys else np.nan,
            "val_pixel": np.asarray(data["genPixelLossValEpochs"], dtype=float) if "genPixelLossValEpochs" in keys else np.nan,
            "train_adv": np.asarray(data["genAdvLossTrainEpochs"], dtype=float) if "genAdvLossTrainEpochs" in keys else np.nan,
            "val_adv": np.asarray(data["genAdvLossValEpochs"], dtype=float) if "genAdvLossValEpochs" in keys else np.nan,
            "train_tv": np.asarray(data["genTvLossTrainEpochs"], dtype=float) if "genTvLossTrainEpochs" in keys else np.nan,
            "val_tv": np.asarray(data["genTvLossValEpochs"], dtype=float) if "genTvLossValEpochs" in keys else np.nan,
            "train_disc_total": np.asarray(data["discLossTrainEpochs"], dtype=float) if "discLossTrainEpochs" in keys else np.nan,
            "val_disc_total": np.asarray(data["discLossValEpochs"], dtype=float) if "discLossValEpochs" in keys else np.nan,
            "train_disc_true": np.asarray(data["discLossTrainTrueEpochs"], dtype=float) if "discLossTrainTrueEpochs" in keys else np.nan,
            "train_disc_fake": np.asarray(data["discLossTrainFakeEpochs"], dtype=float) if "discLossTrainFakeEpochs" in keys else np.nan,
        })
        return df

    # Legacy batch-only lossValues_*.npz fallback: infer number of epochs from filename lossValues_XXXXX.npz
    m_ep = re.search(r"lossValues_(\d+)\.npz$", os.path.basename(latest_npz))
    n_ep = (int(m_ep.group(1)) + 1) if m_ep else 1
    gen_tr_b = np.asarray(data["genLossTrain"], dtype=float) if "genLossTrain" in keys else np.array([])
    gen_val_b = np.asarray(data["genLossVal"], dtype=float) if "genLossVal" in keys else np.array([])
    d_true_b = np.asarray(data["discLossTrain_True"], dtype=float) if "discLossTrain_True" in keys else np.array([])
    d_fake_b = np.asarray(data["discLossTrain_Fake"], dtype=float) if "discLossTrain_Fake" in keys else np.array([])
    d_val_b = np.asarray(data["discLossVal"], dtype=float) if "discLossVal" in keys else np.array([])

    def _chunk_means(arr, k):
        if arr.size == 0:
            return np.full(k, np.nan)
        if arr.size < k:
            k = arr.size
        splits = np.array_split(arr, k)
        return np.array([float(np.mean(s)) for s in splits], dtype=float)

    n_ep = max(1, min(n_ep, max(len(gen_tr_b), len(gen_val_b), 1)))
    tr_gen = _chunk_means(gen_tr_b, n_ep)
    val_gen = _chunk_means(gen_val_b, n_ep)
    tr_dt = _chunk_means(d_true_b, n_ep)
    tr_df = _chunk_means(d_fake_b, n_ep)
    val_d = _chunk_means(d_val_b, n_ep)

    df = pd.DataFrame({
        "epoch": np.arange(1, n_ep + 1, dtype=int),
        "train_gen_total": tr_gen,
        "val_gen_total": val_gen,
        "train_pixel": np.nan,
        "val_pixel": np.nan,
        "train_adv": np.nan,
        "val_adv": np.nan,
        "train_tv": np.nan,
        "val_tv": np.nan,
        "train_disc_total": tr_dt + tr_df,
        "val_disc_total": val_d,
        "train_disc_true": tr_dt,
        "train_disc_fake": tr_df,
    })
    return df


def plot_training_loss_analysis(df_loss: pd.DataFrame, eval_out_dir: str, run_name: str = ""):
    """
    Plot epoch-wise Train vs Validation curves for either Phase 2 Diffusion
    (Total, Noise MSE, x0 L1, Background L1) or Phase 1 cGAN
    (Generator Total, Pixel L1/L2, Adversarial, TV, Discriminator),
    marking the best validation epoch.
    """
    if df_loss is None or df_loss.empty:
        print("[EVAL-LOSS] Kaydedilmiş eğitim kayıp (loss) verisi bulunamadı.")
        return None

    os.makedirs(eval_out_dir, exist_ok=True)
    csv_path = os.path.join(eval_out_dir, "training_loss_history.csv")
    df_loss.to_csv(csv_path, index=False)

    is_diffusion = (
        ("model_type" in df_loss.columns and (df_loss["model_type"] == "diffusion").any())
        or ("train_noise" in df_loss.columns and not df_loss["train_noise"].isna().all())
    )
    epochs = df_loss["epoch"].values

    if is_diffusion:
        val_col = "val_total" if "val_total" in df_loss.columns and not df_loss["val_total"].isna().all() else "val_gen_total"
        tr_col = "train_total" if "train_total" in df_loss.columns and not df_loss["train_total"].isna().all() else "train_gen_total"
        best_idx = int(df_loss[val_col].idxmin())
        best_epoch = int(df_loss.loc[best_idx, "epoch"])
        best_val_loss = float(df_loss.loc[best_idx, val_col])

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        # 1. Diffusion Total Loss
        ax = axes[0]
        ax.plot(epochs, df_loss[tr_col], "o-", label="Train Total Loss", color="#1f77b4", lw=2)
        ax.plot(epochs, df_loss[val_col], "s-", label="Val Total Loss", color="#d62728", lw=2)
        ax.axvline(best_epoch, color="#2ca02c", ls="--", alpha=0.8, label=f"En İyi Epoch: {best_epoch} ({best_val_loss:.4f})")
        ax.scatter([best_epoch], [best_val_loss], color="#2ca02c", s=110, zorder=5, marker="*")
        ax.set_title("Diffusion Total Loss (Train vs Val)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 2. Weighted Noise MSE Loss
        ax_n = axes[1]
        if "train_noise" in df_loss.columns and not df_loss["train_noise"].isna().all():
            ax_n.plot(epochs, df_loss["train_noise"], "o-", label="Train Noise Loss", color="#17becf", lw=2)
            ax_n.plot(epochs, df_loss["val_noise"], "s-", label="Val Noise Loss", color="#e377c2", lw=2)
        ax_n.set_title("Weighted Epsilon-Noise MSE Loss")
        ax_n.set_xlabel("Epoch")
        ax_n.set_ylabel("Noise Loss")
        ax_n.grid(True, alpha=0.3)
        ax_n.legend()

        # 3. x0 Reconstruction Loss
        ax_x0 = axes[2]
        if "train_x0" in df_loss.columns and not df_loss["train_x0"].isna().all():
            ax_x0.plot(epochs, df_loss["train_x0"], "o-", label="Train x0 MSE Loss", color="#9467bd", lw=2)
            ax_x0.plot(epochs, df_loss["val_x0"], "s-", label="Val x0 MSE Loss", color="#8c564b", lw=2)
            if not df_loss["val_x0"].isna().all():
                bx = int(df_loss["val_x0"].idxmin())
                ax_x0.axvline(int(df_loss.loc[bx, "epoch"]), color="#2ca02c", ls=":", alpha=0.8,
                              label=f"Min Val x0 Ep: {int(df_loss.loc[bx, 'epoch'])} ({float(df_loss.loc[bx, 'val_x0']):.4f})")
        ax_x0.set_title("Alpha-Weighted Leaky-Clipped x0 MSE Loss")
        ax_x0.set_xlabel("Epoch")
        ax_x0.set_ylabel("x0 MSE Loss")
        ax_x0.grid(True, alpha=0.3)
        ax_x0.legend()

        # 4. Background L1 Penalty
        ax_bg = axes[3]
        if "train_bg" in df_loss.columns and not df_loss["train_bg"].isna().all():
            ax_bg.plot(epochs, df_loss["train_bg"], "o-", label="Train BG L1", color="#ff7f0e", lw=2)
            ax_bg.plot(epochs, df_loss["val_bg"], "s-", label="Val BG L1", color="#d62728", lw=2)
        ax_bg.set_title("Background Zero-Anchor L1 Penalty")
        ax_bg.set_xlabel("Epoch")
        ax_bg.set_ylabel("BG L1 Loss")
        ax_bg.grid(True, alpha=0.3)
        ax_bg.legend()

        # 5. Train vs Val Gap
        ax_gap = axes[4]
        gap = np.asarray(df_loss[val_col], dtype=float) - np.asarray(df_loss[tr_col], dtype=float)
        ax_gap.plot(epochs, gap, "o-", label="Val - Train Gap", color="#2ca02c", lw=2)
        ax_gap.axhline(0.0, color="#555555", ls="--", lw=1)
        ax_gap.set_title("Generalization Gap (Val - Train Total)")
        ax_gap.set_xlabel("Epoch")
        ax_gap.set_ylabel("Loss Gap")
        ax_gap.grid(True, alpha=0.3)
        ax_gap.legend()

        # 6. Summary text box
        ax_txt = axes[5]
        ax_txt.axis("off")
        summary_lines = [
            f"Run: {run_name} [Diffusion]",
            f"Toplam Epoch Sayısı    : {len(df_loss)}",
            f"En Düşük Val Total Loss: {best_val_loss:.5f} (Ep {best_epoch})",
            f"Son Epoch Train Total  : {df_loss[tr_col].iloc[-1]:.5f}",
            f"Son Epoch Val Total    : {df_loss[val_col].iloc[-1]:.5f}",
        ]
        if "train_noise" in df_loss.columns and not df_loss["train_noise"].isna().all():
            summary_lines.append(f"Son Epoch Train Noise  : {df_loss['train_noise'].iloc[-1]:.5f}")
            summary_lines.append(f"Son Epoch Val Noise    : {df_loss['val_noise'].iloc[-1]:.5f}")
        if "train_x0" in df_loss.columns and not df_loss["train_x0"].isna().all():
            summary_lines.append(f"Son Epoch Train x0     : {df_loss['train_x0'].iloc[-1]:.5f}")
            summary_lines.append(f"Son Epoch Val x0       : {df_loss['val_x0'].iloc[-1]:.5f}")
        if "val_bg" in df_loss.columns and not df_loss["val_bg"].isna().all():
            summary_lines.append(f"Son Epoch Val BG L1    : {df_loss['val_bg'].iloc[-1]:.5f}")
        ax_txt.text(
            0.05, 0.92, "\n".join(summary_lines),
            va="top", ha="left", fontsize=11, family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#cccccc")
        )

        fig.suptitle(f"Diffusion Eğitim Süreci Kayıp (Loss) Analizi — {run_name} (En İyi Epoch: {best_epoch})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig_path = os.path.join(eval_out_dir, "training_loss_curves.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[EVAL-LOSS] Diffusion eğitim kayıp grafiği kaydedildi: {fig_path}")
        print(f"[EVAL-LOSS] En düşük Validation Diffusion Loss: Epoch {best_epoch} (val_total={best_val_loss:.6f})")
        return fig_path

    best_idx = int(df_loss["val_gen_total"].idxmin())
    best_epoch = int(df_loss.loc[best_idx, "epoch"])
    best_val_loss = float(df_loss.loc[best_idx, "val_gen_total"])

    has_components = not df_loss["train_pixel"].isna().all()
    if has_components:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes = np.asarray(axes).ravel()

    # 1. Generator Total Loss
    ax = axes[0]
    ax.plot(epochs, df_loss["train_gen_total"], "o-", label="Train Gen Total", color="#1f77b4", lw=2)
    ax.plot(epochs, df_loss["val_gen_total"], "s-", label="Val Gen Total", color="#d62728", lw=2)
    ax.axvline(best_epoch, color="#2ca02c", ls="--", alpha=0.8, label=f"En İyi Epoch: {best_epoch} ({best_val_loss:.4f})")
    ax.scatter([best_epoch], [best_val_loss], color="#2ca02c", s=110, zorder=5, marker="*")
    ax.set_title("Generator Total Loss (Train vs Val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Discriminator Loss
    ax_d = axes[1]
    ax_d.plot(epochs, df_loss["train_disc_total"], "o-", label="Train Disc Total", color="#9467bd", lw=2)
    if not df_loss["train_disc_true"].isna().all():
        ax_d.plot(epochs, df_loss["train_disc_true"], "^--", label="Train Disc Real", color="#2ca02c", alpha=0.75)
        ax_d.plot(epochs, df_loss["train_disc_fake"], "v--", label="Train Disc Fake", color="#ff7f0e", alpha=0.75)
    ax_d.plot(epochs, df_loss["val_disc_total"], "s-", label="Val Disc Total", color="#8c564b", lw=2)
    ax_d.set_title("Discriminator Loss (Train vs Val)")
    ax_d.set_xlabel("Epoch")
    ax_d.set_ylabel("Loss")
    ax_d.grid(True, alpha=0.3)
    ax_d.legend()

    if has_components:
        # 3. Pixel L1/L2 Loss
        ax_p = axes[2]
        ax_p.plot(epochs, df_loss["train_pixel"], "o-", label="Train Pixel Loss", color="#17becf", lw=2)
        ax_p.plot(epochs, df_loss["val_pixel"], "s-", label="Val Pixel Loss", color="#e377c2", lw=2)
        if not df_loss["val_pixel"].isna().all():
            best_pix_idx = int(df_loss["val_pixel"].idxmin())
            best_pix_ep = int(df_loss.loc[best_pix_idx, "epoch"])
            best_pix_val = float(df_loss.loc[best_pix_idx, "val_pixel"])
            ax_p.axvline(best_pix_ep, color="#2ca02c", ls=":", alpha=0.8, label=f"Min Val Pixel Ep: {best_pix_ep} ({best_pix_val:.4f})")
        ax_p.set_title("Generator Pixel (L1/L2) Loss")
        ax_p.set_xlabel("Epoch")
        ax_p.set_ylabel("Pixel Loss")
        ax_p.grid(True, alpha=0.3)
        ax_p.legend()

        # 4. Adversarial Loss
        ax_a = axes[3]
        ax_a.plot(epochs, df_loss["train_adv"], "o-", label="Train Adversarial", color="#ff7f0e", lw=2)
        ax_a.plot(epochs, df_loss["val_adv"], "s-", label="Val Adversarial", color="#d62728", lw=2)
        ax_a.set_title("Generator Adversarial Loss")
        ax_a.set_xlabel("Epoch")
        ax_a.set_ylabel("Adversarial Loss")
        ax_a.grid(True, alpha=0.3)
        ax_a.legend()

        # 5. Total Variation (TV) Loss
        ax_tv = axes[4]
        ax_tv.plot(epochs, df_loss["train_tv"], "o-", label="Train TV Loss", color="#2ca02c", lw=2)
        ax_tv.plot(epochs, df_loss["val_tv"], "s-", label="Val TV Loss", color="#bcbd22", lw=2)
        ax_tv.set_title("Generator Total Variation (TV) Loss")
        ax_tv.set_xlabel("Epoch")
        ax_tv.set_ylabel("TV Loss")
        ax_tv.grid(True, alpha=0.3)
        ax_tv.legend()

        # 6. Summary text box
        ax_txt = axes[5]
        ax_txt.axis("off")
        summary_lines = [
            f"Run: {run_name}",
            f"Toplam Epoch Sayısı : {len(df_loss)}",
            f"En Düşük Val Gen Loss: {best_val_loss:.5f} (Epoch {best_epoch})",
            f"Son Epoch Train Gen  : {df_loss['train_gen_total'].iloc[-1]:.5f}",
            f"Son Epoch Val Gen    : {df_loss['val_gen_total'].iloc[-1]:.5f}",
            f"Son Epoch Train Pixel: {df_loss['train_pixel'].iloc[-1]:.5f}",
            f"Son Epoch Val Pixel  : {df_loss['val_pixel'].iloc[-1]:.5f}",
            f"Son Epoch Train Disc : {df_loss['train_disc_total'].iloc[-1]:.5f}",
            f"Son Epoch Val Disc   : {df_loss['val_disc_total'].iloc[-1]:.5f}",
        ]
        ax_txt.text(
            0.05, 0.92, "\n".join(summary_lines),
            va="top", ha="left", fontsize=11, family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#cccccc")
        )

    fig.suptitle(f"Eğitim Süreci Kayıp (Loss) Analizi — {run_name} (En İyi Epoch: {best_epoch})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(eval_out_dir, "training_loss_curves.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[EVAL-LOSS] Eğitim kayıp grafiği kaydedildi: {fig_path}")
    print(f"[EVAL-LOSS] En düşük Validation Generator Loss: Epoch {best_epoch} (val_gen_total={best_val_loss:.6f})")
    return fig_path


def build_distance_decay_baseline(gt_dense_raw: np.ndarray) -> np.ndarray:
    """
    Build a symmetric P(d) distance-decay baseline matrix from ground-truth `gt_dense_raw`.
    Each cell (i, j) at genomic distance d = |i - j| is assigned the leave-one-out mean
    interaction value at distance d (or the diagonal mean when diagonal length == 1).
    """
    gt_raw = np.maximum(np.asarray(gt_dense_raw, dtype=np.float64), 0.0)
    n = gt_raw.shape[0]
    base = np.zeros((n, n), dtype=np.float64)
    for d in range(n):
        diag = np.diagonal(gt_raw, offset=d)
        k = diag.size
        if k == 0:
            continue
        if k > 1:
            total = float(np.sum(diag))
            loo_vals = (total - diag) / float(k - 1)
        else:
            loo_vals = np.array([float(diag[0])], dtype=np.float64)
        idx = np.arange(n - d)
        base[idx, idx + d] = loo_vals
        base[idx + d, idx] = loo_vals
    return base


def _compute_composite_score_100(
    auprc_pos: float,
    dist_strat_pearson: float,
    dist_strat_spearman: float,
    pearson_pos_log: float,
    spearman_pos_log: float,
    marg_pearson_log: float,
    marg_spearman_log: float,
    ssim_log: float,
    cos_log: float,
) -> float:
    """
    Combine the 5 core metric groups into a single [0, 100] Composite Model Quality Score:
      - 25%: Active contact capture (AUPRC_GTpos)
      - 25%: Distance-stratified correlation (mean of Pearson & Spearman across d >= 1)
      - 20%: Active-cell intensity correlation (mean of Pearson & Spearman on GT > 0)
      - 15%: 1D marginal hub profile correlation (mean of Pearson & Spearman on row sums)
      - 15%: Structural & geometric similarity (mean of SSIM_log1p & Cosine_log1p)
    """
    def _nz(val):
        if val is None or np.isnan(val):
            return 0.0
        return float(np.clip(val, 0.0, 1.0))

    s_auprc = _nz(auprc_pos)
    s_dist = 0.5 * (_nz(dist_strat_pearson) + _nz(dist_strat_spearman))
    s_active = 0.5 * (_nz(pearson_pos_log) + _nz(spearman_pos_log))
    s_marg = 0.5 * (_nz(marg_pearson_log) + _nz(marg_spearman_log))
    s_struct = 0.5 * (_nz(ssim_log) + _nz(cos_log))

    score = 100.0 * (
        0.25 * s_auprc
        + 0.25 * s_dist
        + 0.20 * s_active
        + 0.15 * s_marg
        + 0.15 * s_struct
    )
    return float(np.clip(score, 0.0, 100.0))


def evaluate_chromosome_pair(
    gt_dense_raw: np.ndarray,
    pred_dense_raw: np.ndarray,
    chrom: str,
    include_baseline: bool = True,
    max_distance_bins: int = None,
) -> dict:
    """
    Compute Hi-C evaluation metrics comparing ground-truth and predicted matrices.
    Pearson & Spearman correlations match the `main` branch (`utils.computePearsonCorrelationSparse`
    & `utils.getCorrelation`): evaluated in RAW counts over the upper-triangular diagonal band
    `0 <= (j - i) < k` (where `k = windowSize` when available), with distance-stratified Pearson AUC
    and Spearman AUC computed via trapezoid rule after dropping constant (NaN) diagonals.
    """
    gt_raw = np.maximum(np.asarray(gt_dense_raw, dtype=np.float64), 0.0)
    pred_raw = np.maximum(np.asarray(pred_dense_raw, dtype=np.float64), 0.0)

    gt_log = np.log1p(gt_raw)
    pred_log = np.log1p(pred_raw)

    n = gt_raw.shape[0]
    k_band = n if (max_distance_bins is None or int(max_distance_bins) <= 0) else min(int(max_distance_bins), n)
    triu_r, triu_c = np.triu_indices(n, k=0)
    band_mask = (triu_c - triu_r) < k_band
    gt_triu_raw = gt_raw[triu_r, triu_c]
    pred_triu_raw = pred_raw[triu_r, triu_c]
    gt_triu_log = gt_log[triu_r, triu_c]
    pred_triu_log = pred_log[triu_r, triu_c]

    gt_band_raw = gt_triu_raw[band_mask]
    pred_band_raw = pred_triu_raw[band_mask]
    gt_band_log = gt_triu_log[band_mask]
    pred_band_log = pred_triu_log[band_mask]

    # Error metrics on the evaluated trapezoid band (matching main's matrixDf region)
    mae_raw = float(np.mean(np.abs(gt_band_raw - pred_band_raw)))
    mse_raw = float(np.mean((gt_band_raw - pred_band_raw) ** 2))
    rmse_raw = float(np.sqrt(mse_raw))

    mae_log = float(np.mean(np.abs(gt_band_log - pred_band_log)))
    mse_log = float(np.mean((gt_band_log - pred_band_log) ** 2))
    rmse_log = float(np.sqrt(mse_log))

    # Overall band correlations in raw and log1p spaces
    pearson_all_raw, spearman_all_raw = safe_corr(gt_band_raw, pred_band_raw)
    pearson_all_log, spearman_all_log = safe_corr(gt_band_log, pred_band_log)

    # Correlations on active bin pairs (GT > 0)
    pos_mask = gt_triu_raw > 0
    n_pos = int(np.sum(pos_mask))
    prevalence_pos = float(n_pos) / float(max(gt_triu_raw.size, 1))
    if n_pos >= 2:
        pearson_pos_raw, spearman_pos_raw = safe_corr(gt_triu_raw[pos_mask], pred_triu_raw[pos_mask])
        pearson_pos_log, spearman_pos_log = safe_corr(gt_triu_log[pos_mask], pred_triu_log[pos_mask])
    else:
        pearson_pos_raw = spearman_pos_raw = pearson_pos_log = spearman_pos_log = float("nan")

    # Distance-stratified correlations & AUC matching main's computePearsonCorrelationSparse (raw counts, d=0..k-1, dropna)
    curves = compute_distance_curves(gt_raw, pred_raw, max_distance_bins=k_band)
    pearson_auc = float(curves["pearson_auc"])
    spearman_auc = float(curves["spearman_auc"])
    dist_strat_pearson_raw = float(curves["mean_dist_pearson_raw"])
    dist_strat_spearman_raw = float(curves["mean_dist_spearman_raw"])
    dist_strat_pearson_log = dist_strat_pearson_raw
    dist_strat_spearman_log = dist_strat_spearman_raw

    # 1D Marginal (row sum) correlations in log1p space
    gt_marg_log = np.sum(gt_log, axis=1)
    pred_marg_log = np.sum(pred_log, axis=1)
    marg_pearson_log, marg_spearman_log = safe_corr(gt_marg_log, pred_marg_log)
    if np.isnan(marg_pearson_log):
        marg_pearson_log = 0.0
    if np.isnan(marg_spearman_log):
        marg_spearman_log = 0.0

    # Structural & similarity metrics
    ssim_raw = compute_ssim_2d(gt_raw, pred_raw)
    psnr_raw = compute_psnr(gt_raw, pred_raw)
    cos_raw = compute_cosine_similarity(gt_band_raw, pred_band_raw)

    ssim_log = compute_ssim_2d(gt_log, pred_log)
    psnr_log = compute_psnr(gt_log, pred_log)
    cos_log = compute_cosine_similarity(gt_band_log, pred_band_log)

    # Binary classification 1: Interaction present (GT > 0)
    y_bin_pos = (gt_triu_raw > 0).astype(np.int32)
    auroc_pos, auprc_pos, f1_pos = compute_binary_classification_metrics(y_bin_pos, pred_triu_log)

    # Binary classification 2: Strong interactions (Top 10% of positive GT contacts, or top 10% overall if dense)
    if n_pos >= 10:
        thresh_top10 = float(np.percentile(gt_triu_raw[pos_mask], 90.0))
        y_bin_top10 = (gt_triu_raw >= thresh_top10).astype(np.int32)
        if np.all(y_bin_top10 == y_bin_pos) and n_pos > 1:
            thresh_top10 = float(np.percentile(gt_triu_raw, 90.0))
            y_bin_top10 = (gt_triu_raw > thresh_top10).astype(np.int32)
    else:
        thresh_top10 = float(np.percentile(gt_triu_raw, 90.0))
        y_bin_top10 = (gt_triu_raw > thresh_top10).astype(np.int32)
    auroc_top10, auprc_top10, f1_top10 = compute_binary_classification_metrics(y_bin_top10, pred_triu_log)

    composite_score = _compute_composite_score_100(
        auprc_pos=auprc_pos,
        dist_strat_pearson=dist_strat_pearson_log,
        dist_strat_spearman=dist_strat_spearman_log,
        pearson_pos_log=0.0 if np.isnan(pearson_pos_log) else pearson_pos_log,
        spearman_pos_log=0.0 if np.isnan(spearman_pos_log) else spearman_pos_log,
        marg_pearson_log=marg_pearson_log,
        marg_spearman_log=marg_spearman_log,
        ssim_log=ssim_log,
        cos_log=cos_log,
    )

    result = {
        "chrom": f"chr{str(chrom).lstrip('chr')}",
        "n_bins": int(n),
        "band_diagonals": int(k_band),
        "gt_nnz_triu": n_pos,
        "Prevalence_GTpos": prevalence_pos,
        "Composite_Score_100": composite_score,
        "AUPRC_GTpos": auprc_pos,
        "AUPRC_Top10pct": auprc_top10,
        "Pearson_AUC": pearson_auc,
        "Spearman_AUC": spearman_auc,
        "DistStrat_Pearson_raw": dist_strat_pearson_raw,
        "DistStrat_Spearman_raw": dist_strat_spearman_raw,
        "DistStrat_Spearman_log1p": dist_strat_spearman_log,
        "DistStrat_Pearson_log1p": dist_strat_pearson_log,
        "Pearson_GTpos_log1p": 0.0 if np.isnan(pearson_pos_log) else pearson_pos_log,
        "Spearman_GTpos_log1p": 0.0 if np.isnan(spearman_pos_log) else spearman_pos_log,
        "Marginal1D_Pearson_log1p": marg_pearson_log,
        "Marginal1D_Spearman_log1p": marg_spearman_log,
        "SSIM_log1p": ssim_log,
        "Cosine_log1p": cos_log,
        "MAE_raw": mae_raw,
        "MSE_raw": mse_raw,
        "RMSE_raw": rmse_raw,
        "MAE_log1p": mae_log,
        "MSE_log1p": mse_log,
        "RMSE_log1p": rmse_log,
        "Pearson_all_raw": pearson_all_raw,
        "Spearman_all_raw": spearman_all_raw,
        "Pearson_all_log1p": pearson_all_log,
        "Spearman_all_log1p": spearman_all_log,
        "Pearson_GTpos_raw": pearson_pos_raw,
        "Spearman_GTpos_raw": spearman_pos_raw,
        "SSIM_raw": ssim_raw,
        "PSNR_raw": psnr_raw,
        "Cosine_raw": cos_raw,
        "PSNR_log1p": psnr_log,
        "AUROC_GTpos": auroc_pos,
        "F1_GTpos": f1_pos,
        "AUROC_Top10pct": auroc_top10,
        "F1_Top10pct": f1_top10,
    }

    if include_baseline:
        base_raw = build_distance_decay_baseline(gt_raw)
        base_metrics = evaluate_chromosome_pair(
            gt_raw, base_raw, chrom=chrom, include_baseline=False, max_distance_bins=k_band
        )
        result["Baseline_Composite_Score_100"] = base_metrics["Composite_Score_100"]
        result["Composite_Delta_vs_Base"] = result["Composite_Score_100"] - base_metrics["Composite_Score_100"]
        result["Baseline_AUPRC_GTpos"] = base_metrics["AUPRC_GTpos"]
        result["AUPRC_Delta_vs_Base"] = result["AUPRC_GTpos"] - base_metrics["AUPRC_GTpos"]
        result["AUPRC_Lift_vs_Random"] = result["AUPRC_GTpos"] / max(prevalence_pos, 1e-6)
        result["Baseline_AUPRC_Top10pct"] = base_metrics["AUPRC_Top10pct"]
        result["Baseline_DistStrat_Spearman_log1p"] = base_metrics["DistStrat_Spearman_log1p"]
        result["Baseline_DistStrat_Pearson_log1p"] = base_metrics["DistStrat_Pearson_log1p"]
        result["Baseline_Pearson_GTpos_log1p"] = base_metrics["Pearson_GTpos_log1p"]
        result["Baseline_Spearman_GTpos_log1p"] = base_metrics["Spearman_GTpos_log1p"]
        result["Baseline_Marginal1D_Pearson_log1p"] = base_metrics["Marginal1D_Pearson_log1p"]
        result["Baseline_SSIM_log1p"] = base_metrics["SSIM_log1p"]
        result["Baseline_Cosine_log1p"] = base_metrics["Cosine_log1p"]

    return result


def compute_distance_curves(
    gt_dense_raw: np.ndarray,
    pred_dense_raw: np.ndarray,
    max_distance_bins: int = None,
):
    """
    Compute distance-dependent P(d) and distance-stratified Pearson/Spearman
    correlations matching the `main` branch (`utils.computePearsonCorrelationSparse`
    & `utils.getCorrelation`):
      - Raw interaction counts (`gt_raw`, `pred_raw`), NOT log1p
      - Diagonals d = 0, 1, ..., k - 1 (where k = max_distance_bins if given, else n)
      - Constant / < 2-element diagonals stay NaN and are omitted (dropna)
      - Computes Pearson AUC & Spearman AUC via sklearn.metrics.auc over normalized d / (k - 1)
    """
    from sklearn import metrics as sk_metrics

    gt_raw = np.maximum(np.asarray(gt_dense_raw, dtype=np.float64), 0.0)
    pred_raw = np.maximum(np.asarray(pred_dense_raw, dtype=np.float64), 0.0)
    gt_log = np.log1p(gt_raw)
    pred_log = np.log1p(pred_raw)

    n = gt_raw.shape[0]
    k = n if (max_distance_bins is None or int(max_distance_bins) <= 0) else min(int(max_distance_bins), n)
    distances = np.arange(0, k, dtype=int)
    gt_pd_raw = np.zeros(k, dtype=np.float64)
    pred_pd_raw = np.zeros(k, dtype=np.float64)
    gt_pd_log = np.zeros(k, dtype=np.float64)
    pred_pd_log = np.zeros(k, dtype=np.float64)
    pearson_by_d = np.full(k, np.nan, dtype=np.float64)
    spearman_by_d = np.full(k, np.nan, dtype=np.float64)

    for d in distances:
        g_diag_raw = np.diagonal(gt_raw, offset=d)
        p_diag_raw = np.diagonal(pred_raw, offset=d)
        g_diag_log = np.diagonal(gt_log, offset=d)
        p_diag_log = np.diagonal(pred_log, offset=d)

        gt_pd_raw[d] = float(np.mean(g_diag_raw))
        pred_pd_raw[d] = float(np.mean(p_diag_raw))
        gt_pd_log[d] = float(np.mean(g_diag_log))
        pred_pd_log[d] = float(np.mean(p_diag_log))

        # Match main's pData.groupby('distance').corr(...) on raw reads (constant diagonals -> NaN)
        pr, sr = safe_corr(g_diag_raw, p_diag_raw)
        if np.isfinite(pr):
            pearson_by_d[d] = float(pr)
        if np.isfinite(sr):
            spearman_by_d[d] = float(sr)

    max_distance = max(int(k - 1), 1)
    mask_p = np.isfinite(pearson_by_d)
    mask_s = np.isfinite(spearman_by_d)

    pearson_auc = (
        float(sk_metrics.auc(distances[mask_p] / float(max_distance), pearson_by_d[mask_p]))
        if int(np.sum(mask_p)) >= 2
        else (float(pearson_by_d[mask_p][0]) if int(np.sum(mask_p)) == 1 else float("nan"))
    )
    spearman_auc = (
        float(sk_metrics.auc(distances[mask_s] / float(max_distance), spearman_by_d[mask_s]))
        if int(np.sum(mask_s)) >= 2
        else (float(spearman_by_d[mask_s][0]) if int(np.sum(mask_s)) == 1 else float("nan"))
    )

    mean_dist_pearson_raw = float(np.mean(pearson_by_d[mask_p])) if np.any(mask_p) else float("nan")
    mean_dist_spearman_raw = float(np.mean(spearman_by_d[mask_s])) if np.any(mask_s) else float("nan")

    return {
        "distances": distances,
        "gt_pd_raw": gt_pd_raw,
        "pred_pd_raw": pred_pd_raw,
        "gt_pd_log": gt_pd_log,
        "pred_pd_log": pred_pd_log,
        "pearson_by_d": pearson_by_d,
        "spearman_by_d": spearman_by_d,
        "pearson_auc": pearson_auc,
        "spearman_auc": spearman_auc,
        "mean_dist_pearson_raw": mean_dist_pearson_raw,
        "mean_dist_spearman_raw": mean_dist_spearman_raw,
        "mean_dist_pearson_log": mean_dist_pearson_raw,
        "mean_dist_spearman_log": mean_dist_spearman_raw,
    }


def render_comparison_panels(
    gt_mat: np.ndarray,
    pred_mat: np.ndarray,
    out_path: str,
    input_mat: np.ndarray = None,
    input_title: str = "Girdi (Input)",
    gt_title: str = "Gerçek (Ground Truth)",
    pred_title: str = "Tahmin Edilen (Predicted)",
    residual_title: str = None,
    suptitle: str = "",
    input_xlabel: str = "Genomik Bin (j)",
    input_ylabel: str = "Genomik Bin (i)",
    matrix_xlabel: str = "Genomik Bin (j)",
    matrix_ylabel: str = "Genomik Bin (i)",
    dpi: int = 300,
):
    """
    Render a unified 4-panel (Input | Ground Truth | Predicted | Residual) or
    3-panel (when input_mat is None) heatmap comparison figure.
    """
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    gt_arr = np.asarray(gt_mat, dtype=np.float64)
    pred_arr = np.asarray(pred_mat, dtype=np.float64)
    residual = pred_arr - gt_arr

    vmax = float(max(np.max(gt_arr), np.max(pred_arr), 1e-6))
    rmax = float(max(np.max(np.abs(residual)), 1e-6))
    mae_l = float(np.mean(np.abs(residual)))
    if residual_title is None:
        residual_title = f"Fark / Residual (Pred - GT) [MAE={mae_l:.4f}]"

    if input_mat is not None:
        inp_arr = np.asarray(input_mat, dtype=np.float64)
        fig, axes = plt.subplots(1, 4, figsize=(24, 5.4))
        inp_aspect = "equal" if (inp_arr.ndim == 2 and inp_arr.shape[0] == inp_arr.shape[1]) else "auto"
        im_in = axes[0].imshow(inp_arr, cmap="viridis", aspect=inp_aspect)
        axes[0].set_title(input_title)
        axes[0].set_xlabel(input_xlabel)
        axes[0].set_ylabel(input_ylabel)
        plt.colorbar(im_in, ax=axes[0], fraction=0.046, pad=0.04)
        ax_gt, ax_pred, ax_res = axes[1], axes[2], axes[3]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
        ax_gt, ax_pred, ax_res = axes[0], axes[1], axes[2]

    im0 = ax_gt.imshow(gt_arr, cmap="Reds", vmin=0.0, vmax=vmax)
    ax_gt.set_title(gt_title)
    ax_gt.set_xlabel(matrix_xlabel)
    ax_gt.set_ylabel(matrix_ylabel)
    plt.colorbar(im0, ax=ax_gt, fraction=0.046, pad=0.04)

    im1 = ax_pred.imshow(pred_arr, cmap="Reds", vmin=0.0, vmax=vmax)
    ax_pred.set_title(pred_title)
    ax_pred.set_xlabel(matrix_xlabel)
    ax_pred.set_ylabel(matrix_ylabel)
    plt.colorbar(im1, ax=ax_pred, fraction=0.046, pad=0.04)

    im2 = ax_res.imshow(residual, cmap="bwr", vmin=-rmax, vmax=rmax)
    ax_res.set_title(residual_title)
    ax_res.set_xlabel(matrix_xlabel)
    ax_res.set_ylabel(matrix_ylabel)
    plt.colorbar(im2, ax=ax_res, fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_triple_heatmaps(chrom_data_dict: dict, eval_out_dir: str, run_name: str = ""):
    """
    Plot 4-panel (Input | Ground Truth | Predicted | Residual) or 3-panel heatmaps
    for each chromosome using log1p(raw) matrices.
    """
    os.makedirs(eval_out_dir, exist_ok=True)
    saved_paths = []
    for chrom, item in chrom_data_dict.items():
        gt_log = np.log1p(item["gt_raw"])
        pred_log = np.log1p(item["pred_raw"])
        mae_l = float(np.mean(np.abs(pred_log - gt_log)))
        input_mat = item.get("input_mat")
        has_input = input_mat is not None
        suptitle = (
            f"chr{chrom} — Girdi, Gerçek, Tahmin ve Fark Haritası ({run_name})"
            if has_input
            else f"chr{chrom} — 3'lü Karşılaştırma ve Residual Haritası ({run_name})"
        )
        out_path = os.path.join(eval_out_dir, f"heatmap_triple_chr{chrom}.png")
        render_comparison_panels(
            gt_mat=gt_log,
            pred_mat=pred_log,
            out_path=out_path,
            input_mat=input_mat,
            input_title=item.get("input_title", f"Girdi (Input) chr{chrom}"),
            gt_title=f"Gerçek (Ground Truth) chr{chrom} [log1p]",
            pred_title=f"Tahmin Edilen (Predicted) chr{chrom} [log1p]",
            residual_title=f"Fark / Residual (Pred - GT) chr{chrom} [MAE={mae_l:.4f}]",
            suptitle=suptitle,
            input_xlabel=item.get("input_xlabel", "Genomik Bin (j)"),
            input_ylabel=item.get("input_ylabel", "Genomik Bin (i)"),
            dpi=300,
        )
        saved_paths.append(out_path)
    return saved_paths


def plot_contact_decay_and_stratified_corr(chrom_data_dict: dict, bin_size: int, eval_out_dir: str, run_name: str = ""):
    """
    Plot:
      1. Genomic distance contact decay curve P(d) (GT vs Predicted, linear & log-log)
      2. Distance-stratified Pearson r and Spearman rho curves across genomic distance d = |i - j|
    """
    os.makedirs(eval_out_dir, exist_ok=True)
    n_chroms = len(chrom_data_dict)
    if n_chroms == 0:
        return []

    # 1. P(d) Contact Decay Curve
    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_chroms, 2)))

    for idx, (chrom, item) in enumerate(chrom_data_dict.items()):
        curves = item["curves"]
        d_mb = curves["distances"] * (bin_size / 1e6)
        c = colors[idx % len(colors)]

        axes1[0].plot(d_mb, curves["gt_pd_log"], "-", color=c, lw=2, label=f"chr{chrom} GT (log1p)")
        axes1[0].plot(d_mb, curves["pred_pd_log"], "--", color=c, lw=2, label=f"chr{chrom} Pred (log1p)")

        valid_d = curves["distances"] >= 1
        axes1[1].plot(
            d_mb[valid_d], np.maximum(curves["gt_pd_raw"][valid_d], 1e-6),
            "-", color=c, lw=2, label=f"chr{chrom} GT (raw)"
        )
        axes1[1].plot(
            d_mb[valid_d], np.maximum(curves["pred_pd_raw"][valid_d], 1e-6),
            "--", color=c, lw=2, label=f"chr{chrom} Pred (raw)"
        )

    axes1[0].set_title("Genomik Mesafeye Göre Etkileşim Şiddeti P(d) [log1p]")
    axes1[0].set_xlabel("Genomik Mesafe |i - j| (Mb)")
    axes1[0].set_ylabel("Ortalama log1p(Etkileşim)")
    axes1[0].grid(True, alpha=0.3)
    axes1[0].legend()

    axes1[1].set_xscale("log")
    axes1[1].set_yscale("log")
    axes1[1].set_title("P(d) Contact Decay Eğrisi [Log-Log Ölçek]")
    axes1[1].set_xlabel("Genomik Mesafe |i - j| (Mb, log)")
    axes1[1].set_ylabel("Ortalama Etkileşim Şiddeti (log)")
    axes1[1].grid(True, which="both", alpha=0.3)
    axes1[1].legend()

    fig1.suptitle(f"Genomik Mesafeye Göre Etkileşim Sönümlenme Eğrisi P(d) — {run_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    pd_path = os.path.join(eval_out_dir, "contact_decay_curve_Pd.png")
    fig1.savefig(pd_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # 2. Distance-Stratified Correlation Curve
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5.5))
    for idx, (chrom, item) in enumerate(chrom_data_dict.items()):
        curves = item["curves"]
        d_mb = curves["distances"] * (bin_size / 1e6)
        c = colors[idx % len(colors)]

        p_vals = curves["pearson_by_d"]
        s_vals = curves["spearman_by_d"]
        mask_p = ~np.isnan(p_vals)
        mask_s = ~np.isnan(s_vals)

        if np.any(mask_p):
            axes2[0].plot(d_mb[mask_p], p_vals[mask_p], "o-", color=c, markersize=3, alpha=0.8, label=f"chr{chrom} (ort={np.nanmean(p_vals):.3f})")
        if np.any(mask_s):
            axes2[1].plot(d_mb[mask_s], s_vals[mask_s], "s-", color=c, markersize=3, alpha=0.8, label=f"chr{chrom} (ort={np.nanmean(s_vals):.3f})")

    axes2[0].axhline(0.0, color="gray", ls="--", lw=1)
    axes2[0].set_title("Mesafeye Göre Katmanlı Pearson Korelasyonu r(d)")
    axes2[0].set_xlabel("Genomik Mesafe |i - j| (Mb)")
    axes2[0].set_ylabel("Pearson r (log1p)")
    axes2[0].set_ylim(-0.3, 1.05)
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend()

    axes2[1].axhline(0.0, color="gray", ls="--", lw=1)
    axes2[1].set_title("Mesafeye Göre Katmanlı Spearman Korelasyonu ρ(d)")
    axes2[1].set_xlabel("Genomik Mesafe |i - j| (Mb)")
    axes2[1].set_ylabel("Spearman ρ (log1p)")
    axes2[1].set_ylim(-0.3, 1.05)
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend()

    fig2.suptitle(f"Genomik Mesafeye Göre Katmanlı (Distance-Stratified) Korelasyon — {run_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    corr_path = os.path.join(eval_out_dir, "distance_stratified_correlation.png")
    fig2.savefig(corr_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    return [pd_path, corr_path]


def plot_stratified_corr(chrom_data_dict: dict, bin_size: int, eval_out_dir: str, run_name: str = ""):
    """Plot distance-stratified Pearson r(d) and Spearman rho(d) curves (raw counts, main style)."""
    os.makedirs(eval_out_dir, exist_ok=True)
    n_chroms = len(chrom_data_dict)
    if n_chroms == 0:
        return None

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_chroms, 2)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for idx, (chrom, item) in enumerate(chrom_data_dict.items()):
        curves = item["curves"]
        d_mb = curves["distances"] * (bin_size / 1e6)
        c = colors[idx % len(colors)]

        p_vals = curves["pearson_by_d"]
        s_vals = curves["spearman_by_d"]
        p_auc = float(curves.get("pearson_auc", np.nan))
        s_auc = float(curves.get("spearman_auc", np.nan))
        mask_p = ~np.isnan(p_vals)
        mask_s = ~np.isnan(s_vals)

        if np.any(mask_p):
            axes[0].plot(
                d_mb[mask_p],
                p_vals[mask_p],
                "o-",
                color=c,
                markersize=3,
                alpha=0.85,
                label=f"chr{chrom} (AUC={p_auc:.3f}, ort={np.nanmean(p_vals):.3f})",
            )
        if np.any(mask_s):
            axes[1].plot(
                d_mb[mask_s],
                s_vals[mask_s],
                "s-",
                color=c,
                markersize=3,
                alpha=0.85,
                label=f"chr{chrom} (AUC={s_auc:.3f}, ort={np.nanmean(s_vals):.3f})",
            )

    axes[0].axhline(0.0, color="gray", ls="--", lw=1)
    axes[0].set_title("Mesafeye Göre Katmanlı Pearson Korelasyonu r(d) [main]")
    axes[0].set_xlabel("Genomik Mesafe |i - j| (Mbp)")
    axes[0].set_ylabel("Pearson r (raw)")
    axes[0].set_ylim(-0.3, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].axhline(0.0, color="gray", ls="--", lw=1)
    axes[1].set_title("Mesafeye Göre Katmanlı Spearman Korelasyonu ρ(d) [main]")
    axes[1].set_xlabel("Genomik Mesafe |i - j| (Mbp)")
    axes[1].set_ylabel("Spearman ρ (raw)")
    axes[1].set_ylim(-0.3, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"Genomik Mesafeye Göre Katmanlı Korelasyon (main yöntemi) — {run_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    corr_path = os.path.join(eval_out_dir, "distance_stratified_correlation.png")
    fig.savefig(corr_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return corr_path


def plot_marginal_1d_profiles(chrom_data_dict: dict, bin_size: int, eval_out_dir: str, run_name: str = ""):
    """
    Plot 1D marginal signal profiles (row sums across each chromosome) for GT vs Predicted.
    """
    os.makedirs(eval_out_dir, exist_ok=True)
    n_chroms = len(chrom_data_dict)
    if n_chroms == 0:
        return None

    fig, axes = plt.subplots(n_chroms, 1, figsize=(15, 4.2 * n_chroms), squeeze=False)
    for idx, (chrom, item) in enumerate(chrom_data_dict.items()):
        ax = axes[idx, 0]
        gt_marg = np.sum(np.log1p(item["gt_raw"]), axis=1)
        pred_marg = np.sum(np.log1p(item["pred_raw"]), axis=1)
        pos_mb = np.arange(len(gt_marg)) * (bin_size / 1e6)

        pr, sr = safe_corr(gt_marg, pred_marg)
        ax.plot(pos_mb, gt_marg, "-", color="#1f77b4", lw=1.8, label="Gerçek (GT) 1B Marjinal Toplam [log1p]")
        ax.plot(pos_mb, pred_marg, "--", color="#d62728", lw=1.8, label="Tahmin (Predicted) 1B Marjinal Toplam [log1p]")
        ax.fill_between(pos_mb, gt_marg, alpha=0.12, color="#1f77b4")
        ax.set_title(
            f"chr{chrom} — 1B Marjinal Sinyal Profili (Bin Bazlı Toplam Etkileşim) | "
            f"Pearson r = {pr:.4f}, Spearman ρ = {sr:.4f}"
        )
        ax.set_xlabel("Genomik Konum (Mb)")
        ax.set_ylabel("Satır Toplamı Σ_j log1p(M_ij)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    fig.suptitle(f"Kromozom Boyunca 1B Marjinal Sinyal Profilleri — {run_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(eval_out_dir, "marginal_1d_profiles.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_value_distribution_hexbin_and_hist(chrom_data_dict: dict, eval_out_dir: str, run_name: str = ""):
    """
    Plot Hexbin scatter plot (GT vs Predicted in log1p space) and marginal density histograms.
    """
    os.makedirs(eval_out_dir, exist_ok=True)
    all_gt_log = []
    all_pred_log = []

    for chrom, item in chrom_data_dict.items():
        n = item["gt_raw"].shape[0]
        r, c = np.triu_indices(n, k=0)
        all_gt_log.append(np.log1p(item["gt_raw"][r, c]))
        all_pred_log.append(np.log1p(item["pred_raw"][r, c]))

    if not all_gt_log:
        return None

    gt_vec = np.concatenate(all_gt_log)
    pred_vec = np.concatenate(all_pred_log)
    pos_mask = gt_vec > 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # 1. Hexbin on all upper-triangle bin pairs
    hb0 = axes[0].hexbin(gt_vec, pred_vec, gridsize=35, cmap="viridis", bins="log", mincnt=1)
    max_v = float(max(gt_vec.max(), pred_vec.max(), 1.0))
    axes[0].plot([0, max_v], [0, max_v], "r--", lw=1.5, label="y = x (İdeal)")
    pr_all, sr_all = safe_corr(gt_vec, pred_vec)
    axes[0].set_title(f"Tüm Bin Çiftleri Hexbin [log1p]\n(r={pr_all:.4f}, ρ={sr_all:.4f})")
    axes[0].set_xlabel("Gerçek (GT) [log1p]")
    axes[0].set_ylabel("Tahmin (Predicted) [log1p]")
    axes[0].legend(loc="upper left")
    plt.colorbar(hb0, ax=axes[0], label="log10(Bin Çifti Sayısı)")

    # 2. Hexbin on active (GT > 0) bin pairs
    if np.sum(pos_mask) >= 2:
        hb1 = axes[1].hexbin(gt_vec[pos_mask], pred_vec[pos_mask], gridsize=30, cmap="magma", bins="log", mincnt=1)
        pr_pos, sr_pos = safe_corr(gt_vec[pos_mask], pred_vec[pos_mask])
        axes[1].plot([0, max_v], [0, max_v], "w--", lw=1.5, label="y = x (İdeal)")
        axes[1].set_title(f"Aktif Etkileşimler (GT > 0) Hexbin\n(r={pr_pos:.4f}, ρ={sr_pos:.4f}, n={int(np.sum(pos_mask))})")
        axes[1].set_xlabel("Gerçek (GT > 0) [log1p]")
        axes[1].set_ylabel("Tahmin (Predicted) [log1p]")
        axes[1].legend(loc="upper left")
        plt.colorbar(hb1, ax=axes[1], label="log10(Bin Çifti Sayısı)")
    else:
        axes[1].text(0.5, 0.5, "GT > 0 aktif bin çifti yeterli değil", ha="center", va="center")

    # 3. Value Distribution Histogram (Log-density)
    axes[2].hist(gt_vec, bins=40, alpha=0.55, color="#1f77b4", label="Gerçek (GT) [log1p]", log=True)
    axes[2].hist(pred_vec, bins=40, alpha=0.55, color="#d62728", label="Tahmin (Predicted) [log1p]", log=True)
    axes[2].set_title("Gerçek vs Tahmin Değer Dağılımı (Log-Frekans)")
    axes[2].set_xlabel("Değer [log1p]")
    axes[2].set_ylabel("Bin Çifti Sayısı (log ölçek)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle(f"Değer Dağılımı ve Hexbin Saçılım Analizi — {run_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(eval_out_dir, "value_distribution_hexbin_hist.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _diagnose_score_row(row: pd.Series) -> str:
    """Return a concise diagnostic status badge comparing the model row against the P(d) baseline."""
    chrom_label = str(row.get("chrom", ""))
    if "STD" in chrom_label:
        return "± Değişkenlik"
    comp = float(row.get("Composite_Score_100", 0.0) or 0.0)
    delta_comp = float(row.get("Composite_Delta_vs_Base", 0.0) or 0.0)
    dist_rho = float(row.get("DistStrat_Spearman_log1p", 0.0) or 0.0)
    auprc_delta = float(row.get("AUPRC_Delta_vs_Base", 0.0) or 0.0)

    if comp >= 45.0 or (delta_comp >= 5.0 and dist_rho >= 0.15):
        return "🟢 Güçlü (Baseline Üstü Yapısal Öğrenme)"
    if delta_comp >= 0.0 or dist_rho >= 0.05 or auprc_delta >= 0.0:
        return "🟡 Orta (P(d) Yakalandı / İnce Yapı Gelişiyor)"
    return "🟠 Başlangıç (Daha Fazla Epoch/Kromozom Önerilir)"


def build_core_scorecard_df(df_metrics_full: pd.DataFrame) -> pd.DataFrame:
    """
    Build a concise paper metrics scorecard table aligned with `main` branch Pearson/Spearman
    calculation (raw counts over the evaluated diagonal band, distance-stratified AUC & mean r(d)/rho(d),
    plus MAE, MSE, RMSE, SSIM, PSNR).
    """
    if df_metrics_full is None or df_metrics_full.empty:
        return pd.DataFrame()

    rows = []
    for _, r in df_metrics_full.iterrows():
        rows.append({
            "Kromozom": r.get("chrom"),
            "Pearson AUC (main)": float(r.get("Pearson_AUC", np.nan)),
            "Spearman AUC (main)": float(r.get("Spearman_AUC", np.nan)),
            "Katmanlı Pearson r(d) (raw)": float(r.get("DistStrat_Pearson_raw", np.nan)),
            "Katmanlı Spearman ρ(d) (raw)": float(r.get("DistStrat_Spearman_raw", np.nan)),
            "Band Pearson r (raw)": float(r.get("Pearson_all_raw", np.nan)),
            "Band Spearman ρ (raw)": float(r.get("Spearman_all_raw", np.nan)),
            "MAE (raw)": float(r.get("MAE_raw", np.nan)),
            "MSE (raw)": float(r.get("MSE_raw", np.nan)),
            "RMSE (raw)": float(r.get("RMSE_raw", np.nan)),
            "SSIM (log1p)": float(r.get("SSIM_log1p", np.nan)),
            "PSNR (log1p)": float(r.get("PSNR_log1p", np.nan)),
        })
    return pd.DataFrame(rows)


def style_core_scorecard(df_scorecard: pd.DataFrame):
    """Apply readable Pandas Styler formatting and color highlights to the Core Scorecard DataFrame."""
    if df_scorecard is None or df_scorecard.empty:
        return df_scorecard

    str_cols = {"Kromozom", "Model Tanı / Yorum", "Run Adı", "Kromozomlar", "Girdi Modu", "Pencere Modu"}
    num_cols = [c for c in df_scorecard.columns if c not in str_cols and pd.api.types.is_numeric_dtype(df_scorecard[c])]
    fmt_dict = {c: "{:.4f}" for c in num_cols}
    for int_col in ("Sıra", "WS", "Epoch"):
        if int_col in fmt_dict:
            fmt_dict[int_col] = "{:.0f}"
    for score_col in ("Bileşik Skor (0-100)", "Baz P(d) Skor (0-100)", "Baz Skor (0-100)"):
        if score_col in fmt_dict:
            fmt_dict[score_col] = "{:.2f}"
    if "Skor Farkı (Δ vs Baz)" in fmt_dict:
        fmt_dict["Skor Farkı (Δ vs Baz)"] = "{:+.2f}"
    if "AUPRC Katı (vs Rastgele)" in fmt_dict:
        fmt_dict["AUPRC Katı (vs Rastgele)"] = "{:.2f}x"

    def _color_delta(val):
        try:
            v = float(val)
            if np.isnan(v):
                return ""
            if v > 0.5:
                return "background-color: #d4edda; color: #155724; font-weight: bold;"
            if v >= -0.5:
                return "background-color: #fff3cd; color: #856404;"
            return "background-color: #f8d7da; color: #721c24;"
        except Exception:
            return ""

    try:
        styler = df_scorecard.style.format(fmt_dict, na_rep="-")
        if "Skor Farkı (Δ vs Baz)" in df_scorecard.columns:
            styler = styler.map(_color_delta, subset=["Skor Farkı (Δ vs Baz)"])
        return styler
    except (AttributeError, ImportError):
        df_fmt = df_scorecard.copy()
        for col in num_cols:
            if col in ("Bileşik Skor (0-100)", "Baz P(d) Skor (0-100)", "Skor Farkı (Δ vs Baz)", "AUPRC Katı (vs Rastgele)"):
                df_fmt[col] = df_fmt[col].round(2)
            elif col == "Sıra":
                df_fmt[col] = df_fmt[col].astype(int)
            else:
                df_fmt[col] = df_fmt[col].round(4)
        return df_fmt


def plot_model_vs_baseline_radar(df_metrics_full: pd.DataFrame, eval_out_dir: str, run_name: str = ""):
    """
    Plot a 2-panel visual summary comparing the Trained Model against the P(d) Distance-Decay Baseline:
      - Left: 6-axis Polar Radar (Spider) Chart of normalized [0, 1] core metrics
      - Right: Grouped Horizontal Bar Chart of Model vs P(d) Baseline with Delta callouts
    """
    if df_metrics_full is None or df_metrics_full.empty:
        return None

    os.makedirs(eval_out_dir, exist_ok=True)
    mean_rows = df_metrics_full[df_metrics_full["chrom"].astype(str).str.startswith("MEAN")]
    row = mean_rows.iloc[0] if not mean_rows.empty else df_metrics_full.iloc[0]

    def _clip01(v):
        if v is None or pd.isna(v):
            return 0.0
        return float(np.clip(float(v), 0.0, 1.0))

    labels = [
        "1) AUPRC\n(GT > 0)",
        "1b) AUPRC\n(Top %10)",
        "2) Katmanlı\nSpearman ρ(d)",
        "3) Aktif Hücre\nPearson (GT>0)",
        "4) 1B Marjinal\nProfil r",
        "5) Yapısal\nSSIM (log1p)",
    ]
    model_vals = [
        _clip01(row.get("AUPRC_GTpos", 0.0)),
        _clip01(row.get("AUPRC_Top10pct", 0.0)),
        _clip01(row.get("DistStrat_Spearman_log1p", 0.0)),
        _clip01(row.get("Pearson_GTpos_log1p", 0.0)),
        _clip01(row.get("Marginal1D_Pearson_log1p", 0.0)),
        _clip01(row.get("SSIM_log1p", 0.0)),
    ]
    base_vals = [
        _clip01(row.get("Baseline_AUPRC_GTpos", 0.0)),
        _clip01(row.get("Baseline_AUPRC_Top10pct", 0.0)),
        _clip01(row.get("Baseline_DistStrat_Spearman_log1p", 0.0)),
        _clip01(row.get("Baseline_Pearson_GTpos_log1p", 0.0)),
        _clip01(row.get("Baseline_Marginal1D_Pearson_log1p", 0.0)),
        _clip01(row.get("Baseline_SSIM_log1p", 0.0)),
    ]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]
    model_closed = model_vals + [model_vals[0]]
    base_closed = base_vals + [base_vals[0]]

    fig = plt.figure(figsize=(16, 6.2))
    ax_radar = fig.add_subplot(1, 2, 1, polar=True)
    ax_bar = fig.add_subplot(1, 2, 2)

    # Panel 1: Radar Chart
    ax_radar.plot(angles_closed, model_closed, "o-", lw=2.4, color="#d62728", label="Eğitilen Hi-cGAN Modeli")
    ax_radar.fill(angles_closed, model_closed, color="#d62728", alpha=0.22)
    ax_radar.plot(angles_closed, base_closed, "s--", lw=2.0, color="#1f77b4", label="P(d) Mesafe Ortalaması Baz Modeli")
    ax_radar.fill(angles_closed, base_closed, color="#1f77b4", alpha=0.12)

    ax_radar.set_thetagrids(np.degrees(angles), labels, fontsize=9.5)
    max_r = max(max(model_vals), max(base_vals), 0.5)
    ax_radar.set_ylim(0.0, min(1.0, max_r * 1.2 + 0.05))
    ax_radar.set_title("Çekirdek Metrikler Radar Profili [0-1 Ölçeği]", pad=18, fontsize=12, fontweight="bold")
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))

    # Panel 2: Grouped Horizontal Bar Comparison
    bar_names = [
        "Bileşik Skor (/100)",
        "AUPRC (GT > 0) [x100]",
        "AUPRC (Top %10) [x100]",
        "Katmanlı ρ(d) [x100]",
        "Aktif Hücre r [x100]",
        "1B Marjinal r [x100]",
        "SSIM (log1p) [x100]",
    ]
    comp_m = float(row.get("Composite_Score_100", 0.0) or 0.0)
    comp_b = float(row.get("Baseline_Composite_Score_100", 0.0) or 0.0)
    m_bars = [comp_m] + [v * 100.0 for v in model_vals]
    b_bars = [comp_b] + [v * 100.0 for v in base_vals]

    y_pos = np.arange(len(bar_names))
    height = 0.36
    ax_bar.barh(y_pos + height / 2, m_bars, height=height, color="#d62728", alpha=0.88, label="Eğitilen Model")
    ax_bar.barh(y_pos - height / 2, b_bars, height=height, color="#1f77b4", alpha=0.68, label="P(d) Baz Modeli")

    for i, (mv, bv) in enumerate(zip(m_bars, b_bars)):
        diff = mv - bv
        sign = "+" if diff >= 0 else ""
        ax_bar.text(
            max(mv, bv) + 1.2, i,
            f"{mv:.1f} ({sign}{diff:.1f} vs Baz)",
            va="center", ha="left", fontsize=9, fontweight="bold" if i == 0 else "normal"
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_names, fontsize=10)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, max(max(m_bars), max(b_bars), 50.0) * 1.32)
    ax_bar.set_xlabel("Skor (0 - 100 Ölçeği)")
    ax_bar.set_title(
        f"Model vs P(d) Baz Model Kıyaslaması | Bileşik Skor: {comp_m:.2f}/100 (Baz: {comp_b:.2f})",
        fontsize=11.5, fontweight="bold"
    )
    ax_bar.grid(True, axis="x", alpha=0.3)
    ax_bar.legend(loc="lower right")

    fig.suptitle(f"Çekirdek 5 Metrik Karnesi ve Baseline Analizi — {run_name}", fontsize=13.5, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(eval_out_dir, "core_metrics_radar_and_baseline.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def evaluate_predictions_dir(
    pred_npz_dir: str,
    gt_npz_dir: str,
    chroms=None,
    eval_out_dir: str = None,
    bin_size: int = 250000,
    run_name: str = "rna-cgan-clean",
    make_plots: bool = True,
    chromatin_dir: str = None,
    feature_set: str = "representative",
    use_pair_features: bool = True,
):
    """
    Unified Referee Evaluation for a directory of predicted chromosome `.npz` files (`pred_npz_dir`)
    against ground-truth `.npz` files (`gt_npz_dir`).
    Computes the Core 5 Metrics + P(d) Distance-Decay Baseline + 0-100 Composite Score for each chromosome,
    appends MEAN and STD summary rows, saves `evaluation_metrics_full.csv` and `core_metrics_scorecard.csv`
    in `eval_out_dir`, and optionally generates diagnostic plots.
    Returns `(df_scorecard, df_metrics_full)`.
    """
    from . import utils, featurePack

    if eval_out_dir is None:
        eval_out_dir = os.path.join(os.path.dirname(os.path.normpath(pred_npz_dir)), "evaluation")
    os.makedirs(eval_out_dir, exist_ok=True)

    has_feature_pack = bool(chromatin_dir and featurePack.is_feature_pack_source(chromatin_dir))
    eval_window_size = _resolve_eval_window_size(pred_npz_dir)

    npz_files = sorted(glob.glob(os.path.join(pred_npz_dir, "chr*.npz")))
    chrom_filter = {str(c).lstrip("chr") for c in chroms} if chroms else None

    metric_rows = []
    chrom_data_dict = {}
    for pfile in npz_files:
        chrom = os.path.splitext(os.path.basename(pfile))[0].lstrip("chr")
        if chrom_filter is not None and chrom not in chrom_filter:
            continue
        gt_sp, _ = utils.getMatrixFromNpz(gt_npz_dir, chrom, pBinSizeInt=bin_size)
        pr_sp = sparse.load_npz(pfile)
        gt_dense = np.asarray(gt_sp.toarray(), dtype=np.float64)
        pr_dense = np.asarray(pr_sp.toarray(), dtype=np.float64)
        if gt_dense.shape != pr_dense.shape:
            log.warning("Shape mismatch for chr%s: GT=%s vs Pred=%s", chrom, gt_dense.shape, pr_dense.shape)
            continue

        input_mat = None
        input_title = f"Girdi (Input) chr{chrom}"
        input_xlabel = "Genomik Bin (j)"
        input_ylabel = "Genomik Bin (i)"
        if has_feature_pack:
            try:
                n_bins = int(gt_dense.shape[0])
                pair_sp = None
                if use_pair_features:
                    try:
                        pair_sp = featurePack.get_chrom_pair_matrix(chromatin_dir, chrom, pNumBins=n_bins)
                    except Exception as pair_exc:
                        log.debug("2D pair features unavailable for chr%s, falling back to 1D: %s", chrom, str(pair_exc))
                        pair_sp = None
                if pair_sp is not None and pair_sp.nnz > 0:
                    input_mat = np.asarray(pair_sp.toarray(), dtype=np.float64)
                    input_title = f"Girdi (2D kmer7_revcomp_cosine) chr{chrom}"
                else:
                    feat_arr, _ = featurePack.get_chrom_bin_features(
                        chromatin_dir, chrom, pFeatureSet=feature_set, pNumBins=n_bins
                    )
                    if feat_arr is not None and feat_arr.size > 0:
                        input_mat = np.asarray(feat_arr, dtype=np.float64)
                        input_title = f"Girdi (1D {feature_set}, {feat_arr.shape[1]} ch) chr{chrom}"
                        input_xlabel = "Feature Kanalı"
            except Exception as exc:
                log.warning("Could not load input features for chr%s: %s", chrom, str(exc))

        row = evaluate_chromosome_pair(gt_dense, pr_dense, chrom=chrom, include_baseline=False, max_distance_bins=eval_window_size)
        metric_rows.append(row)
        chrom_data_dict[chrom] = {
            "gt_raw": gt_dense,
            "pred_raw": pr_dense,
            "input_mat": input_mat,
            "input_title": input_title,
            "input_xlabel": input_xlabel,
            "input_ylabel": input_ylabel,
            "curves": compute_distance_curves(gt_dense, pr_dense, max_distance_bins=eval_window_size),
        }

    if not metric_rows:
        return pd.DataFrame(), pd.DataFrame()

    df_chroms = pd.DataFrame(metric_rows)
    num_cols = [c for c in df_chroms.columns if c != "chrom" and pd.api.types.is_numeric_dtype(df_chroms[c])]
    if len(df_chroms) > 1:
        mean_row = {"chrom": f"MEAN ({len(df_chroms)} chroms)"}
        std_row = {"chrom": f"STD ({len(df_chroms)} chroms)"}
        for col in num_cols:
            mean_row[col] = float(df_chroms[col].mean())
            std_row[col] = float(df_chroms[col].std(ddof=0))
        df_metrics_full = pd.concat([df_chroms, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    else:
        df_metrics_full = df_chroms

    df_scorecard = build_core_scorecard_df(df_metrics_full)

    full_csv = os.path.join(eval_out_dir, "evaluation_metrics_full.csv")
    scorecard_csv = os.path.join(eval_out_dir, "core_metrics_scorecard.csv")
    df_metrics_full.to_csv(full_csv, index=False)
    df_scorecard.to_csv(scorecard_csv, index=False)

    if make_plots:
        plot_calls = [
            ("triple_heatmaps", lambda: plot_triple_heatmaps(chrom_data_dict, eval_out_dir, run_name=run_name)),
            ("distance_stratified_correlation", lambda: plot_stratified_corr(chrom_data_dict, bin_size, eval_out_dir, run_name=run_name)),
        ]
        for plot_name, fn in plot_calls:
            try:
                fn()
            except Exception as exc:
                log.warning("Could not generate evaluation plot '%s': %s", plot_name, str(exc))
                print(f"[EVAL-PLOT-WARN] {plot_name}: {exc}", flush=True)

    return df_scorecard, df_metrics_full


def _detect_model_type(run_dir: str, train_dir: str = None, df_loss: pd.DataFrame = None) -> str:
    """Detect whether a run directory belongs to Diffusion or cGAN."""
    import pickle
    if df_loss is not None and not df_loss.empty:
        if "model_type" in df_loss.columns and (df_loss["model_type"] == "diffusion").any():
            return "Diffusion"
        if "train_noise" in df_loss.columns and not df_loss["train_noise"].isna().all():
            return "Diffusion"
    cand_dirs = [d for d in (train_dir, run_dir) if d and os.path.isdir(d)]
    for d in cand_dirs:
        if os.path.exists(os.path.join(d, "denoiser_best.keras")) or os.path.exists(os.path.join(d, "denoiser_final.keras")):
            return "Diffusion"
        meta_p = os.path.join(d, "hicgan_metadata.pkl")
        if os.path.exists(meta_p):
            try:
                with open(meta_p, "rb") as mf:
                    mobj = pickle.load(mf)
                if isinstance(mobj, dict):
                    mt = str(mobj.get("modelType", "")).lower()
                    if mt == "diffusion" or "timesteps" in mobj:
                        return "Diffusion"
                    if mt == "cgan":
                        return "cGAN"
            except Exception:
                pass
        tlog = os.path.join(d, "training.log")
        if os.path.exists(tlog):
            try:
                with open(tlog, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read(65536)
                if "HiCGAN-Diffusion" in txt or "denoiser_best.keras" in txt or "train_noise=" in txt:
                    return "Diffusion"
            except Exception:
                pass
    rlow = os.path.basename(os.path.normpath(run_dir)).lower()
    if "diff" in rlow:
        return "Diffusion"
    return "cGAN"


def compare_runs_leaderboard(
    search_root_dirs,
    npz_data_dir: str,
    bin_size: int = 250000,
    chroms=None,
    eval_out_dir: str = None,
    max_runs: int = 8,
    include_baseline_row: bool = True,
) -> pd.DataFrame:
    """
    Scan run directories in `search_root_dirs` (e.g. ['output_models', 'notebook_pipeline_output']),
    compute or load Core 5 Metrics + Composite Score (0-100) for each run that has predicted `.npz`
    chromosome matrices, rank them (plus optional P(d) Baseline row) in a Leaderboard DataFrame,
    and plot a Multi-Run Radar & Bar chart.
    """
    from . import utils

    if isinstance(search_root_dirs, str):
        search_root_dirs = [search_root_dirs]

    chrom_filter = {str(c).lstrip("chr") for c in chroms} if chroms else None

    def _resolve_npz_and_train_dirs(d):
        for cand_npz in (
            os.path.join(d, "predict_run", "npz_matrices"),
            os.path.join(d, "test_eval", "npz_matrices"),
            os.path.join(d, "npz_matrices"),
            d,
        ):
            files = sorted(glob.glob(os.path.join(cand_npz, "chr*.npz")))
            if files:
                if os.path.isdir(os.path.join(d, "train_run")):
                    t_dir = os.path.join(d, "train_run")
                elif os.path.basename(os.path.normpath(d)) in ("predict_run", "test_eval", "npz_matrices"):
                    parent = os.path.dirname(os.path.normpath(d))
                    t_dir = os.path.join(parent, "train_run") if os.path.isdir(os.path.join(parent, "train_run")) else parent
                else:
                    t_dir = d
                return cand_npz, t_dir, files
        return None, None, []

    run_dirs = []
    seen_real = set()
    for root in search_root_dirs:
        if not root or not os.path.exists(root):
            continue
        _, _, direct_files = _resolve_npz_and_train_dirs(root)
        if (
            direct_files
            or os.path.isdir(os.path.join(root, "predict_run"))
            or os.path.isdir(os.path.join(root, "test_eval"))
            or os.path.isdir(os.path.join(root, "train_run"))
        ):
            candidates = [root]
        else:
            candidates = sorted(glob.glob(os.path.join(root, "run_*")))
            if not candidates and os.path.isdir(root):
                candidates = [os.path.join(root, sub) for sub in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, sub))]
        for cdir in candidates:
            rp = os.path.realpath(cdir)
            if rp not in seen_real and os.path.isdir(cdir):
                seen_real.add(rp)
                run_dirs.append(cdir)

    leaderboard_rows = []
    baseline_ref_chroms = []
    for rdir in run_dirs:
        rname = os.path.basename(os.path.normpath(rdir))
        if rname in ("npz_matrices", "predict_run", "test_eval"):
            rname = os.path.basename(os.path.dirname(os.path.normpath(rdir))) or rname
        pred_npz_dir, train_dir, npz_files = _resolve_npz_and_train_dirs(rdir)
        if not npz_files:
            continue

        # Parse run metadata from directory name, hicgan_metadata.pkl, or predictions.pkl
        m_name = re.match(r"run_\d+_\d+_([^_]+)_([^_]+)_ws(\d+)_ep(\d+)", rname)
        input_mode = m_name.group(1) if m_name else "?"
        window_mode = m_name.group(2) if m_name else "?"
        ws = int(m_name.group(3)) if m_name else 64
        ep = int(m_name.group(4)) if m_name else 0

        import pickle
        for meta_cand in (
            os.path.join(train_dir, "hicgan_metadata.pkl"),
            os.path.join(rdir, "hicgan_metadata.pkl"),
            os.path.join(os.path.dirname(pred_npz_dir), "predictions.pkl"),
            os.path.join(rdir, "predictions.pkl"),
        ):
            if os.path.exists(meta_cand):
                try:
                    with open(meta_cand, "rb") as mf:
                        mobj = pickle.load(mf)
                    if isinstance(mobj, dict):
                        if input_mode == "?" and mobj.get("inputMode"):
                            input_mode = str(mobj["inputMode"])
                        if window_mode == "?" and mobj.get("windowMode"):
                            window_mode = str(mobj["windowMode"])
                        if mobj.get("windowSize") is not None:
                            ws = int(mobj["windowSize"])
                except Exception:
                    pass

        # Check best val loss from training history if available
        df_loss = parse_training_losses(train_dir)
        if df_loss.empty and rdir != train_dir:
            df_loss = parse_training_losses(rdir)
        model_type = _detect_model_type(rdir, train_dir=train_dir, df_loss=df_loss)

        if not df_loss.empty:
            if ep == 0:
                ep = int(len(df_loss))
            if "val_total" in df_loss and not df_loss["val_total"].isna().all():
                best_val_loss = float(df_loss["val_total"].min())
            elif "val_pixel" in df_loss and not df_loss["val_pixel"].isna().all():
                best_val_loss = float(df_loss["val_pixel"].min())
            elif "val_gen_total" in df_loss and not df_loss["val_gen_total"].isna().all():
                best_val_loss = float(df_loss["val_gen_total"].min())
            else:
                best_val_loss = np.nan
        else:
            best_val_loss = np.nan

        chrom_rows = []
        chrom_names = []
        for pfile in npz_files:
            chrom = os.path.splitext(os.path.basename(pfile))[0].lstrip("chr")
            if chrom_filter is not None and chrom not in chrom_filter:
                continue
            try:
                gt_sp, _ = utils.getMatrixFromNpz(npz_data_dir, chrom, pBinSizeInt=bin_size)
                pr_sp = sparse.load_npz(pfile)
                if gt_sp.shape == pr_sp.shape:
                    crow = evaluate_chromosome_pair(
                        gt_sp.toarray(), pr_sp.toarray(), chrom=chrom, include_baseline=False, max_distance_bins=ws
                    )
                    chrom_rows.append(crow)
                    chrom_names.append(f"chr{chrom}")
            except Exception as exc:
                log.debug("Skipping %s in %s: %s", pfile, rname, str(exc))

        if not chrom_rows:
            continue

        dfc = pd.DataFrame(chrom_rows)
        leaderboard_rows.append({
            "Run Adı": rname,
            "Model Tipi": model_type,
            "Girdi Modu": input_mode,
            "Pencere Modu": window_mode,
            "WS": ws,
            "Epoch": ep,
            "Kromozomlar": ",".join(chrom_names),
            "Pearson AUC (main)": float(dfc["Pearson_AUC"].mean()),
            "Spearman AUC (main)": float(dfc["Spearman_AUC"].mean()),
            "Katmanlı Pearson r(d) (raw)": float(dfc["DistStrat_Pearson_raw"].mean()),
            "Katmanlı Spearman ρ(d) (raw)": float(dfc["DistStrat_Spearman_raw"].mean()),
            "Band Pearson r (raw)": float(dfc["Pearson_all_raw"].mean()),
            "Band Spearman ρ (raw)": float(dfc["Spearman_all_raw"].mean()),
            "MAE (raw)": float(dfc["MAE_raw"].mean()),
            "MSE (raw)": float(dfc["MSE_raw"].mean()),
            "RMSE (raw)": float(dfc["RMSE_raw"].mean()),
            "SSIM (log1p)": float(dfc["SSIM_log1p"].mean()),
            "PSNR (log1p)": float(dfc["PSNR_log1p"].mean()),
            "Min Val Loss": best_val_loss,
        })

    if not leaderboard_rows:
        return pd.DataFrame()

    df_lb = pd.DataFrame(leaderboard_rows).sort_values("Pearson AUC (main)", ascending=False).reset_index(drop=True)
    df_lb.insert(0, "Sıra", np.arange(1, len(df_lb) + 1))

    if eval_out_dir:
        os.makedirs(eval_out_dir, exist_ok=True)
        lb_csv = os.path.join(eval_out_dir, "multi_run_leaderboard.csv")
        df_lb.to_csv(lb_csv, index=False)

        df_plot = df_lb.head(max_runs)
        y_pos = np.arange(len(df_plot))
        short_names = [f"#{int(r['Sıra'])} {str(r['Run Adı']).replace('run_', '')}" for _, r in df_plot.iterrows()]

        fig, ax = plt.subplots(figsize=(10, max(3.6, 0.55 * len(df_plot) + 1.4)))
        bar_h = 0.36
        pearson_vals = df_plot["Pearson AUC (main)"].values
        spearman_vals = df_plot["Spearman AUC (main)"].values

        ax.barh(y_pos - bar_h / 2, pearson_vals, height=bar_h, color="#1f77b4", alpha=0.88, label="Pearson AUC (main)")
        ax.barh(y_pos + bar_h / 2, spearman_vals, height=bar_h, color="#ff7f0e", alpha=0.88, label="Spearman AUC (main)")
        for i, (pv, sv) in enumerate(zip(pearson_vals, spearman_vals)):
            ax.text(max(pv, 0.0) + 0.015, i - bar_h / 2, f"{pv:.3f}", va="center", ha="left", fontsize=8.5)
            ax.text(max(sv, 0.0) + 0.015, i + bar_h / 2, f"{sv:.3f}", va="center", ha="left", fontsize=8.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names, fontsize=9)
        ax.invert_yaxis()
        max_x = max(float(np.nanmax(pearson_vals)), float(np.nanmax(spearman_vals)), 0.5)
        ax.set_xlim(min(0.0, float(np.nanmin(pearson_vals)), float(np.nanmin(spearman_vals))) - 0.05, max_x * 1.18)
        ax.set_xlabel("Korelasyon AUC Skoru (main yöntemi)")
        ax.set_title("Run Liderlik Sıralaması (Pearson & Spearman AUC — main)", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(loc="lower right")
        plt.tight_layout()
        bar_path = os.path.join(eval_out_dir, "multi_run_comparison.png")
        fig.savefig(bar_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return df_lb


def _resolve_model_run_artifacts(run_or_npz_dir: str, chroms=None):
    """
    Given a run root dir (containing train_run / predict_run / test_eval / npz_matrices)
    or an npz_matrices / predict_run / test_eval dir,
    resolve (chrom_to_npz_path, train_dir, predict_dir, metadata_dict).
    """
    import json
    import pickle

    d = os.path.abspath(run_or_npz_dir)
    npz_dir = None
    for cand in (
        os.path.join(d, "predict_run", "npz_matrices"),
        os.path.join(d, "test_eval", "npz_matrices"),
        os.path.join(d, "npz_matrices"),
        d,
    ):
        if glob.glob(os.path.join(cand, "chr*.npz")):
            npz_dir = cand
            break
    if npz_dir is None:
        raise FileNotFoundError(f"Predicted chr*.npz matrices not found under: {run_or_npz_dir}")

    predict_dir = os.path.dirname(npz_dir) if os.path.basename(npz_dir) == "npz_matrices" else d
    parent_of_pred = os.path.dirname(predict_dir)
    if os.path.isdir(os.path.join(d, "train_run")):
        train_dir = os.path.join(d, "train_run")
    elif os.path.isdir(os.path.join(parent_of_pred, "train_run")):
        train_dir = os.path.join(parent_of_pred, "train_run")
    elif any(
        os.path.exists(os.path.join(d, fn))
        for fn in ("hicgan_metadata.pkl", "training.log", "target_value_range.json")
    ):
        train_dir = d
    elif os.path.basename(predict_dir) in ("predict_run", "test_eval") and any(
        os.path.exists(os.path.join(parent_of_pred, fn))
        for fn in ("hicgan_metadata.pkl", "training.log", "target_value_range.json")
    ):
        train_dir = parent_of_pred
    else:
        train_dir = d

    chrom_filter = {str(c).lstrip("chr") for c in chroms} if chroms else None
    chrom_map = {}
    for pfile in sorted(glob.glob(os.path.join(npz_dir, "chr*.npz"))):
        chrom = os.path.splitext(os.path.basename(pfile))[0].lstrip("chr")
        if chrom_filter is not None and chrom not in chrom_filter:
            continue
        chrom_map[chrom] = pfile

    meta = {}
    for pkl_cand in (
        os.path.join(train_dir, "hicgan_metadata.pkl"),
        os.path.join(d, "hicgan_metadata.pkl"),
        os.path.join(predict_dir, "predictions.pkl"),
        os.path.join(d, "predictions.pkl"),
    ):
        if os.path.exists(pkl_cand):
            try:
                with open(pkl_cand, "rb") as fh:
                    obj = pickle.load(fh)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k not in meta and v is not None:
                            meta[k] = v
            except Exception:
                pass

    for json_cand in (
        os.path.join(train_dir, "target_value_range.json"),
        os.path.join(d, "target_value_range.json"),
    ):
        if os.path.exists(json_cand):
            try:
                with open(json_cand, "r", encoding="utf-8") as fh:
                    jobj = json.load(fh)
                if isinstance(jobj, dict):
                    if "targetNorm" not in meta and jobj.get("method"):
                        meta["targetNorm"] = jobj.get("method")
                    if "target_range_hi" not in meta and jobj.get("hi", jobj.get("max")) is not None:
                        meta["target_range_hi"] = float(jobj.get("hi", jobj.get("max")))
                    if "target_range_scale_factor" not in meta and jobj.get("scale_factor") is not None:
                        meta["target_range_scale_factor"] = float(jobj.get("scale_factor"))
            except Exception:
                pass

    return {
        "root_dir": d,
        "npz_dir": npz_dir,
        "predict_dir": predict_dir,
        "train_dir": train_dir,
        "chrom_map": chrom_map,
        "meta": meta,
    }


def _audit_comparison_fairness(cgan_info: dict, diff_info: dict, common_chroms: list, bin_size: int) -> dict:
    """
    Audit configuration parity between Phase 1 cGAN and Phase 2 Diffusion runs
    so any discrepancy in splits, binSize, windowSize, gridStride, normalization,
    SAT filtering, blendMode, dropout, or feature representation is explicitly logged.
    """
    m1 = cgan_info.get("meta", {})
    m2 = diff_info.get("meta", {})
    fields = [
        "binSize",
        "windowSize",
        "windowMode",
        "gridStride",
        "inputMode",
        "featureSet",
        "usePairFeatures",
        "filterEmptyWindows",
        "targetNorm",
        "logpNorm",
        "target_range_hi",
        "lossMode",
        "logTransformTarget",
        "flankPadChromosomes",
        "flankPadMaxBins",
        "flankPadMinRatio",
        "blendMode",
        "dropout",
        "seed",
    ]
    float_fields = {"target_range_hi", "dropout", "flankPadMinRatio"}
    checks = {}
    warnings = []
    for k in fields:
        v1 = m1.get(k)
        v2 = m2.get(k)
        if v1 is None and v2 is None:
            status = "missing_both"
        elif v1 is None or v2 is None:
            status = "partial"
            warnings.append(f"Field '{k}' only recorded in one run: cGAN={v1!r}, Diffusion={v2!r}")
        else:
            if k in float_fields:
                try:
                    is_eq = bool(np.isclose(float(v1), float(v2), rtol=1e-4, atol=1e-6))
                except Exception:
                    is_eq = (v1 == v2)
            else:
                is_eq = (v1 == v2)
            if is_eq:
                status = "match"
            else:
                status = "mismatch"
                warnings.append(f"Mismatch on '{k}': cGAN={v1!r} vs Diffusion={v2!r}")
        checks[k] = {"cgan": v1, "diffusion": v2, "status": status}

    c_only = sorted(set(cgan_info["chrom_map"].keys()) - set(diff_info["chrom_map"].keys()))
    d_only = sorted(set(diff_info["chrom_map"].keys()) - set(cgan_info["chrom_map"].keys()))
    if c_only or d_only:
        warnings.append(
            f"Predicted chromosome sets differ (cGAN-only={c_only}, Diff-only={d_only}); "
            f"strictly evaluating on intersection={common_chroms}."
        )

    has_mismatch = any(v["status"] == "mismatch" for v in checks.values())
    return {
        "fair_comparison": (not has_mismatch) and bool(common_chroms),
        "bin_size_requested": int(bin_size),
        "evaluated_chromosomes": [f"chr{c}" for c in common_chroms],
        "cgan_only_chroms": [f"chr{c}" for c in c_only],
        "diffusion_only_chroms": [f"chr{c}" for c in d_only],
        "field_checks": checks,
        "warnings": warnings,
    }


def compare_two_models(
    cgan_dir: str,
    diffusion_dir: str,
    npz_data_dir: str,
    eval_out_dir: str,
    bin_size: int = 250000,
    chroms=None,
    cgan_name: str = "Phase 1: cGAN (rna-cgan-clean)",
    diffusion_name: str = "Phase 2: Diffusion (rna-diffusion-clean)",
    baseline_name: str = "Baseline: P(d) Mesafe Ortalaması",
    chromatin_dir: str = None,
    feature_set: str = "representative",
    use_pair_features: bool = True,
    max_heatmap_chroms: int = 4,
) -> dict:
    """
    Fair head-to-head comparison between Phase 1 cGAN (`cgan_dir`),
    Phase 2 Diffusion (`diffusion_dir`), and the `P(d)` Distance-Decay Baseline
    on the exact chromosome intersection.

    Writes:
      - `fairness_audit.json`
      - `per_chromosome_comparison.csv`
      - `comparison_leaderboard.csv`
      - `comparison_radar_and_bars.png`
      - `comparison_per_chrom_scores.png`
      - `comparison_distance_curves.png`
      - `comparison_heatmaps_chr{c}.png` (for up to `max_heatmap_chroms`)
    """
    import json
    from . import utils, featurePack

    os.makedirs(eval_out_dir, exist_ok=True)
    has_feature_pack = bool(chromatin_dir and featurePack.is_feature_pack_source(chromatin_dir))
    cgan_info = _resolve_model_run_artifacts(cgan_dir, chroms=chroms)
    diff_info = _resolve_model_run_artifacts(diffusion_dir, chroms=chroms)

    common_chroms = sorted(
        set(cgan_info["chrom_map"].keys()) & set(diff_info["chrom_map"].keys()),
        key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
    )
    if not common_chroms:
        raise ValueError(
            f"No overlapping predicted chr*.npz chromosomes found between cGAN ({cgan_info['npz_dir']}) "
            f"and Diffusion ({diff_info['npz_dir']})."
        )

    fairness = _audit_comparison_fairness(cgan_info, diff_info, common_chroms, bin_size=bin_size)
    fairness_path = os.path.join(eval_out_dir, "fairness_audit.json")
    with open(fairness_path, "w", encoding="utf-8") as fh:
        json.dump(fairness, fh, indent=2, default=str)

    df_loss_cgan = parse_training_losses(cgan_info["train_dir"])
    df_loss_diff = parse_training_losses(diff_info["train_dir"])

    def _best_ep_and_loss(dfl):
        if dfl is None or dfl.empty:
            return 0, np.nan
        col = "val_total" if ("val_total" in dfl.columns and not dfl["val_total"].isna().all()) else (
            "val_pixel" if ("val_pixel" in dfl.columns and not dfl["val_pixel"].isna().all()) else "val_gen_total"
        )
        if col not in dfl.columns or dfl[col].isna().all():
            return int(len(dfl)), np.nan
        idx = int(dfl[col].idxmin())
        return int(dfl.loc[idx, "epoch"]), float(dfl.loc[idx, col])

    cgan_best_ep, cgan_best_loss = _best_ep_and_loss(df_loss_cgan)
    diff_best_ep, diff_best_loss = _best_ep_and_loss(df_loss_diff)

    cmp_ws = (
        cgan_info["meta"].get("windowSize")
        or diff_info["meta"].get("windowSize")
        or _resolve_eval_window_size(cgan_info["npz_dir"])
        or _resolve_eval_window_size(diff_info["npz_dir"])
    )
    cmp_ws = int(cmp_ws) if cmp_ws else None

    per_chrom_rows = []
    cgan_chrom_rows = []
    diff_chrom_rows = []
    dist_curves_cgan = []
    dist_curves_diff = []
    heatmap_files = []

    core_cols = [
        "Pearson_AUC",
        "Spearman_AUC",
        "DistStrat_Pearson_raw",
        "DistStrat_Spearman_raw",
        "Pearson_all_raw",
        "Spearman_all_raw",
        "MAE_raw",
        "MSE_raw",
        "RMSE_raw",
        "SSIM_log1p",
        "PSNR_log1p",
    ]

    for idx_c, chrom in enumerate(common_chroms):
        gt_sp, _ = utils.getMatrixFromNpz(npz_data_dir, chrom, pBinSizeInt=bin_size)
        cgan_sp = sparse.load_npz(cgan_info["chrom_map"][chrom])
        diff_sp = sparse.load_npz(diff_info["chrom_map"][chrom])

        if gt_sp.shape != cgan_sp.shape or gt_sp.shape != diff_sp.shape:
            raise ValueError(
                f"Shape mismatch on chr{chrom}: GT={gt_sp.shape}, cGAN={cgan_sp.shape}, Diffusion={diff_sp.shape}"
            )

        gt_dense = gt_sp.toarray()
        cgan_dense = cgan_sp.toarray()
        diff_dense = diff_sp.toarray()

        row_cgan = evaluate_chromosome_pair(
            gt_dense, cgan_dense, chrom=chrom, include_baseline=False, max_distance_bins=cmp_ws
        )
        row_diff = evaluate_chromosome_pair(
            gt_dense, diff_dense, chrom=chrom, include_baseline=False, max_distance_bins=cmp_ws
        )
        cgan_chrom_rows.append(row_cgan)
        diff_chrom_rows.append(row_diff)

        merged = {"Chromosome": f"chr{chrom}", "N_bins": int(gt_dense.shape[0])}
        for k in core_cols:
            merged[f"cGAN_{k}"] = float(row_cgan[k])
            merged[f"Diffusion_{k}"] = float(row_diff[k])
            merged[f"Delta_Diff_minus_cGAN_{k}"] = float(row_diff[k] - row_cgan[k])
        per_chrom_rows.append(merged)

        dist_curves_cgan.append(compute_distance_curves(gt_dense, cgan_dense, max_distance_bins=cmp_ws))
        dist_curves_diff.append(compute_distance_curves(gt_dense, diff_dense, max_distance_bins=cmp_ws))

        if idx_c < max_heatmap_chroms:
            input_mat = None
            input_title = f"Girdi (Input) chr{chrom}"
            input_xlabel = "Bin"
            input_ylabel = "Bin"
            if has_feature_pack:
                try:
                    n_bins = int(gt_dense.shape[0])
                    pair_sp = None
                    if use_pair_features:
                        try:
                            pair_sp = featurePack.get_chrom_pair_matrix(chromatin_dir, chrom, pNumBins=n_bins)
                        except Exception as pair_exc:
                            log.debug("2D pair features unavailable for chr%s in comparison, falling back to 1D: %s", chrom, str(pair_exc))
                            pair_sp = None
                    if pair_sp is not None and pair_sp.nnz > 0:
                        input_mat = np.asarray(pair_sp.toarray(), dtype=np.float64)
                        input_title = f"Girdi (Input)\n2D kmer7_revcomp_cosine chr{chrom}"
                    else:
                        feat_arr, _ = featurePack.get_chrom_bin_features(
                            chromatin_dir, chrom, pFeatureSet=feature_set, pNumBins=n_bins
                        )
                        if feat_arr is not None and feat_arr.size > 0:
                            input_mat = np.asarray(feat_arr, dtype=np.float64)
                            input_title = f"Girdi (Input)\n1D {feature_set} ({feat_arr.shape[1]} ch) chr{chrom}"
                            input_xlabel = "Feature Kanalı"
                except Exception as exc:
                    log.warning("Could not load comparison input features for chr%s: %s", chrom, str(exc))

            fig = None
            try:
                gt_log = np.log1p(np.maximum(gt_dense, 0.0))
                cgan_log = np.log1p(np.maximum(cgan_dense, 0.0))
                diff_log = np.log1p(np.maximum(diff_dense, 0.0))
                vmax = float(np.percentile(gt_log[gt_log > 0], 99.5)) if np.any(gt_log > 0) else 1.0
                vmax = max(vmax, 1e-3)

                if input_mat is not None:
                    fig, axes = plt.subplots(1, 5, figsize=(24, 4.4))
                    inp_arr = np.asarray(input_mat, dtype=np.float64)
                    inp_aspect = "equal" if (inp_arr.ndim == 2 and inp_arr.shape[0] == inp_arr.shape[1]) else "auto"
                    im_in = axes[0].imshow(inp_arr, cmap="viridis", aspect=inp_aspect, origin="upper", interpolation="nearest")
                    axes[0].set_title(input_title, fontsize=10, fontweight="bold")
                    axes[0].set_xlabel(input_xlabel)
                    axes[0].set_ylabel(input_ylabel)
                    plt.colorbar(im_in, ax=axes[0], fraction=0.046, pad=0.04)
                    mat_axes = axes[1:4]
                    diff_ax = axes[4]
                else:
                    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
                    mat_axes = axes[:3]
                    diff_ax = axes[3]

                panels = [
                    (gt_log, f"chr{chrom} Ground Truth (log1p)", "YlOrRd", 0.0, vmax),
                    (cgan_log, f"cGAN\nAUC_P={row_cgan['Pearson_AUC']:.3f}, AUC_S={row_cgan['Spearman_AUC']:.3f}", "YlOrRd", 0.0, vmax),
                    (diff_log, f"Diffusion\nAUC_P={row_diff['Pearson_AUC']:.3f}, AUC_S={row_diff['Spearman_AUC']:.3f}", "YlOrRd", 0.0, vmax),
                ]
                for ax, (mat, ttl, cm, v0, v1) in zip(mat_axes, panels):
                    im = ax.imshow(mat, cmap=cm, vmin=v0, vmax=v1, origin="upper", interpolation="nearest")
                    ax.set_title(ttl, fontsize=10, fontweight="bold")
                    ax.set_xlabel("Bin")
                    ax.set_ylabel("Bin")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                diff_map = diff_log - cgan_log
                dlim = max(float(np.percentile(np.abs(diff_map), 99.5)), 1e-3)
                im_d = diff_ax.imshow(diff_map, cmap="coolwarm", vmin=-dlim, vmax=dlim, origin="upper", interpolation="nearest")
                delta_auc = row_diff["Pearson_AUC"] - row_cgan["Pearson_AUC"]
                diff_ax.set_title(f"Diff - cGAN (log1p)\nΔAUC_P={delta_auc:+.4f}", fontsize=10, fontweight="bold")
                diff_ax.set_xlabel("Bin")
                plt.colorbar(im_d, ax=diff_ax, fraction=0.046, pad=0.04)

                fig.suptitle(f"Model Karşılaştırma Isı Haritası — chr{chrom} ({bin_size // 1000} kb)", fontsize=12.5, fontweight="bold")
                plt.tight_layout()
                hm_path = os.path.join(eval_out_dir, f"comparison_heatmaps_chr{chrom}.png")
                fig.savefig(hm_path, dpi=250, bbox_inches="tight")
                heatmap_files.append(hm_path)
            except Exception as exc:
                log.warning("Could not generate comparison heatmap for chr%s: %s", chrom, str(exc))
                print(f"[COMPARE-PLOT-WARN] comparison_heatmaps_chr{chrom}: {exc}", flush=True)
            finally:
                if fig is not None:
                    plt.close(fig)

    df_per_chrom = pd.DataFrame(per_chrom_rows)
    per_chrom_csv = os.path.join(eval_out_dir, "per_chromosome_comparison.csv")
    df_per_chrom.to_csv(per_chrom_csv, index=False)

    df_c = pd.DataFrame(cgan_chrom_rows)
    df_d = pd.DataFrame(diff_chrom_rows)
    cgan_pauc_mean = float(df_c["Pearson_AUC"].mean())
    diff_pauc_mean = float(df_d["Pearson_AUC"].mean())
    cgan_sauc_mean = float(df_c["Spearman_AUC"].mean())
    diff_sauc_mean = float(df_d["Spearman_AUC"].mean())
    cgan_pearson_mean = float(df_c["Pearson_all_raw"].mean())
    diff_pearson_mean = float(df_d["Pearson_all_raw"].mean())
    cgan_spearman_mean = float(df_c["Spearman_all_raw"].mean())
    diff_spearman_mean = float(df_d["Spearman_all_raw"].mean())
    chrom_label = ",".join(f"chr{c}" for c in common_chroms)

    def _lb_row(name, mtype, dff, ep, vloss):
        return {
            "Model Adı": name,
            "Model Tipi": mtype,
            "Kromozomlar": chrom_label,
            "Best Epoch": ep,
            "Min Val Loss": vloss,
            "Pearson AUC (main)": float(dff["Pearson_AUC"].mean()),
            "Spearman AUC (main)": float(dff["Spearman_AUC"].mean()),
            "Katmanlı Pearson r(d) (raw)": float(dff["DistStrat_Pearson_raw"].mean()),
            "Katmanlı Spearman ρ(d) (raw)": float(dff["DistStrat_Spearman_raw"].mean()),
            "Band Pearson r (raw)": float(dff["Pearson_all_raw"].mean()),
            "Band Spearman ρ (raw)": float(dff["Spearman_all_raw"].mean()),
            "MAE (raw)": float(dff["MAE_raw"].mean()),
            "MSE (raw)": float(dff["MSE_raw"].mean()),
            "RMSE (raw)": float(dff["RMSE_raw"].mean()),
            "SSIM (log1p)": float(dff["SSIM_log1p"].mean()),
            "PSNR (log1p)": float(dff["PSNR_log1p"].mean()),
        }

    df_lb = pd.DataFrame([
        _lb_row(diffusion_name, "Diffusion", df_d, diff_best_ep, diff_best_loss),
        _lb_row(cgan_name, "cGAN", df_c, cgan_best_ep, cgan_best_loss),
    ]).sort_values("Pearson AUC (main)", ascending=False).reset_index(drop=True)
    df_lb.insert(0, "Sıra", np.arange(1, len(df_lb) + 1))
    lb_csv = os.path.join(eval_out_dir, "comparison_leaderboard.csv")
    df_lb.to_csv(lb_csv, index=False)

    # Plot 1: Paper Metrics Grouped Bar Chart (no polar radar)
    radar_path = None
    fig_bar = None
    try:
        fig_bar, (ax_corr, ax_err) = plt.subplots(1, 2, figsize=(15, 5.2))
        corr_keys = [
            "Pearson AUC (main)",
            "Spearman AUC (main)",
            "Katmanlı Pearson r(d) (raw)",
            "Katmanlı Spearman ρ(d) (raw)",
            "Band Pearson r (raw)",
            "Band Spearman ρ (raw)",
            "SSIM (log1p)",
        ]
        y_c = np.arange(len(corr_keys))
        h = 0.36
        cgan_row = df_lb[df_lb["Model Tipi"] == "cGAN"].iloc[0]
        diff_row = df_lb[df_lb["Model Tipi"] == "Diffusion"].iloc[0]

        c_vals = [float(cgan_row[k]) for k in corr_keys]
        d_vals = [float(diff_row[k]) for k in corr_keys]
        ax_corr.barh(y_c - h / 2, c_vals, height=h, color="#ff7f0e", alpha=0.88, label="cGAN")
        ax_corr.barh(y_c + h / 2, d_vals, height=h, color="#1f77b4", alpha=0.88, label="Diffusion")
        for i, (cv, dv) in enumerate(zip(c_vals, d_vals)):
            ax_corr.text(max(cv, 0.0) + 0.015, i - h / 2, f"{cv:.3f}", va="center", ha="left", fontsize=8.5)
            ax_corr.text(max(dv, 0.0) + 0.015, i + h / 2, f"{dv:.3f}", va="center", ha="left", fontsize=8.5)
        ax_corr.set_yticks(y_c)
        ax_corr.set_yticklabels(corr_keys, fontsize=9)
        ax_corr.invert_yaxis()
        ax_corr.set_xlim(min(0.0, float(np.nanmin(c_vals + d_vals))) - 0.05, max(max(c_vals + d_vals), 0.5) * 1.18)
        ax_corr.set_xlabel("Skor")
        ax_corr.set_title("Korelasyon (main) ve Yapısal Benzerlik Metrikleri (Yüksek = İyi)", fontsize=11, fontweight="bold")
        ax_corr.grid(True, axis="x", alpha=0.3)
        ax_corr.legend(loc="lower right")

        err_keys = ["MAE (raw)", "RMSE (raw)"]
        x_e = np.arange(len(err_keys))
        w_e = 0.34
        ce_vals = [float(cgan_row[k]) for k in err_keys]
        de_vals = [float(diff_row[k]) for k in err_keys]
        ax_err.bar(x_e - w_e / 2, ce_vals, width=w_e, color="#ff7f0e", alpha=0.88, label="cGAN")
        ax_err.bar(x_e + w_e / 2, de_vals, width=w_e, color="#1f77b4", alpha=0.88, label="Diffusion")
        for i, (cv, dv) in enumerate(zip(ce_vals, de_vals)):
            ax_err.text(i - w_e / 2, cv * 1.02, f"{cv:.4f}", ha="center", va="bottom", fontsize=8.5)
            ax_err.text(i + w_e / 2, dv * 1.02, f"{dv:.4f}", ha="center", va="bottom", fontsize=8.5)
        ax_err.set_xticks(x_e)
        ax_err.set_xticklabels(err_keys, fontsize=9.5)
        ax_err.set_ylabel("Hata (raw)")
        ax_err.set_title("Hata Metrikleri (Düşük = İyi)", fontsize=11, fontweight="bold")
        ax_err.grid(True, axis="y", alpha=0.3)
        ax_err.legend()

        fig_bar.suptitle(
            f"Faz 1 cGAN vs Faz 2 Diffusion (Δ Pearson AUC = {diff_pauc_mean - cgan_pauc_mean:+.4f})",
            fontsize=12.5, fontweight="bold",
        )
        plt.tight_layout()
        cand_radar = os.path.join(eval_out_dir, "comparison_metrics_bars.png")
        fig_bar.savefig(cand_radar, dpi=250, bbox_inches="tight")
        radar_path = cand_radar
    except Exception as exc:
        log.warning("Could not generate comparison plot 'comparison_metrics_bars': %s", str(exc))
        print(f"[COMPARE-PLOT-WARN] comparison_metrics_bars: {exc}", flush=True)
    finally:
        if fig_bar is not None:
            plt.close(fig_bar)

    # Plot 2: Per-chromosome Pearson AUC & Spearman AUC grouped bars (main method)
    per_chrom_png = None
    fig_pc = None
    try:
        fig_pc, (ax_pr, ax_sr) = plt.subplots(1, 2, figsize=(15, 5.2))
        x = np.arange(len(df_per_chrom))
        w = 0.35
        ax_pr.bar(x - w / 2, df_per_chrom["cGAN_Pearson_AUC"], width=w, color="#ff7f0e", label="cGAN")
        ax_pr.bar(x + w / 2, df_per_chrom["Diffusion_Pearson_AUC"], width=w, color="#1f77b4", label="Diffusion")
        ax_pr.set_xticks(x)
        ax_pr.set_xticklabels(df_per_chrom["Chromosome"])
        ax_pr.set_ylabel("Pearson AUC (main)")
        ax_pr.set_title("Kromozom Bazlı Pearson AUC (main)")
        ax_pr.grid(True, axis="y", alpha=0.3)
        ax_pr.legend()

        ax_sr.bar(x - w / 2, df_per_chrom["cGAN_Spearman_AUC"], width=w, color="#ff7f0e", label="cGAN")
        ax_sr.bar(x + w / 2, df_per_chrom["Diffusion_Spearman_AUC"], width=w, color="#1f77b4", label="Diffusion")
        ax_sr.set_xticks(x)
        ax_sr.set_xticklabels(df_per_chrom["Chromosome"])
        ax_sr.set_ylabel("Spearman AUC (main)")
        ax_sr.set_title("Kromozom Bazlı Spearman AUC (main)")
        ax_sr.grid(True, axis="y", alpha=0.3)
        ax_sr.legend()

        plt.tight_layout()
        cand_pc = os.path.join(eval_out_dir, "comparison_per_chrom_scores.png")
        fig_pc.savefig(cand_pc, dpi=250, bbox_inches="tight")
        per_chrom_png = cand_pc
    except Exception as exc:
        log.warning("Could not generate comparison plot 'comparison_per_chrom_scores': %s", str(exc))
        print(f"[COMPARE-PLOT-WARN] comparison_per_chrom_scores: {exc}", flush=True)
    finally:
        if fig_pc is not None:
            plt.close(fig_pc)

    # Plot 3: Distance-Stratified Pearson r(d) & Spearman rho(d) Curves (raw counts, main style)
    dist_png = None
    fig_dist = None
    try:
        min_len = min(min(len(dc["distances"]) for dc in dist_curves_cgan), min(len(dc["distances"]) for dc in dist_curves_diff))
        min_len = min(min_len, 80)
        if min_len >= 2:
            dists = dist_curves_cgan[0]["distances"][:min_len]
            d_mb = dists * (bin_size / 1e6)

            def _safe_nanmean(arrays):
                stacked = np.vstack(arrays)
                out = np.full(stacked.shape[1], np.nan, dtype=np.float64)
                valid = ~np.all(np.isnan(stacked), axis=0)
                if np.any(valid):
                    out[valid] = np.nanmean(stacked[:, valid], axis=0)
                return out

            cgan_pr = _safe_nanmean([dc["pearson_by_d"][:min_len] for dc in dist_curves_cgan])
            diff_pr = _safe_nanmean([dc["pearson_by_d"][:min_len] for dc in dist_curves_diff])
            cgan_rho = _safe_nanmean([dc["spearman_by_d"][:min_len] for dc in dist_curves_cgan])
            diff_rho = _safe_nanmean([dc["spearman_by_d"][:min_len] for dc in dist_curves_diff])

            fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
            mp_c = ~np.isnan(cgan_pr)
            mp_d = ~np.isnan(diff_pr)
            ms_c = ~np.isnan(cgan_rho)
            ms_d = ~np.isnan(diff_rho)
            ax1.plot(d_mb[mp_c], cgan_pr[mp_c], "--", color="#ff7f0e", lw=2.0, label=f"Phase 1: cGAN (AUC={cgan_pauc_mean:.3f})")
            ax1.plot(d_mb[mp_d], diff_pr[mp_d], "-", color="#1f77b4", lw=2.0, label=f"Phase 2: Diffusion (AUC={diff_pauc_mean:.3f})")
            ax1.axhline(0.0, color="#666666", lw=0.9, ls=":")
            ax1.set_xlabel("Genomik Mesafe |i - j| (Mbp)")
            ax1.set_ylabel("Pearson r(d) [raw]")
            ax1.set_title("Mesafeye Göre Katmanlı Pearson r(d) [main]")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.plot(d_mb[ms_c], cgan_rho[ms_c], "--", color="#ff7f0e", lw=2.0, label=f"Phase 1: cGAN (AUC={cgan_sauc_mean:.3f})")
            ax2.plot(d_mb[ms_d], diff_rho[ms_d], "-", color="#1f77b4", lw=2.0, label=f"Phase 2: Diffusion (AUC={diff_sauc_mean:.3f})")
            ax2.axhline(0.0, color="#666666", lw=0.9, ls=":")
            ax2.set_xlabel("Genomik Mesafe |i - j| (Mbp)")
            ax2.set_ylabel("Spearman ρ(d) [raw]")
            ax2.set_title("Mesafeye Göre Katmanlı Spearman ρ(d) [main]")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            plt.tight_layout()
            cand_dist = os.path.join(eval_out_dir, "comparison_distance_curves.png")
            fig_dist.savefig(cand_dist, dpi=250, bbox_inches="tight")
            dist_png = cand_dist
    except Exception as exc:
        log.warning("Could not generate comparison plot 'comparison_distance_curves': %s", str(exc))
        print(f"[COMPARE-PLOT-WARN] comparison_distance_curves: {exc}", flush=True)
    finally:
        if fig_dist is not None:
            plt.close(fig_dist)

    print(f"[COMPARE] Adil Karşılaştırma tamamlandı -> {eval_out_dir}")
    print(f"[COMPARE] Fairness check: fair_comparison={fairness['fair_comparison']} | warnings={len(fairness['warnings'])}")
    print(df_lb.to_string(index=False))

    return {
        "leaderboard": df_lb,
        "per_chromosome": df_per_chrom,
        "fairness_audit": fairness,
        "summary_metrics": {
            "cgan_pearson_auc": cgan_pauc_mean,
            "diffusion_pearson_auc": diff_pauc_mean,
            "delta_pearson_auc": diff_pauc_mean - cgan_pauc_mean,
            "cgan_spearman_auc": cgan_sauc_mean,
            "diffusion_spearman_auc": diff_sauc_mean,
            "delta_spearman_auc": diff_sauc_mean - cgan_sauc_mean,
            "cgan_pearson_raw": cgan_pearson_mean,
            "diffusion_pearson_raw": diff_pearson_mean,
            "delta_pearson_raw": diff_pearson_mean - cgan_pearson_mean,
            "cgan_spearman_raw": cgan_spearman_mean,
            "diffusion_spearman_raw": diff_spearman_mean,
            "delta_spearman_raw": diff_spearman_mean - cgan_spearman_mean,
        },
        "files": {
            "fairness_audit": fairness_path,
            "leaderboard_csv": lb_csv,
            "per_chromosome_csv": per_chrom_csv,
            "radar_bar_png": radar_path,
            "per_chrom_png": per_chrom_png,
            "distance_curves_png": dist_png,
            "heatmap_pngs": heatmap_files,
        },
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare Phase 1 cGAN and Phase 2 Diffusion predictions fairly.")
    parser.add_argument("--cganDir", required=True, help="cGAN run directory or predict_run/npz_matrices directory")
    parser.add_argument("--diffusionDir", required=True, help="Diffusion run directory or predict_run/npz_matrices directory")
    parser.add_argument("--gtDir", required=True, help="Ground-truth .npz matrices directory")
    parser.add_argument("--outDir", required=True, help="Output directory for comparison CSVs, JSON audit, and plots")
    parser.add_argument("--chroms", nargs="*", default=None, help="Optional chromosome filter (e.g. 2 6 12)")
    parser.add_argument("--binSize", type=int, default=250000, help="Bin size in bp (default: 250000)")
    parser.add_argument("--cganName", default="Phase 1: cGAN (rna-cgan-clean)", help="Display name for cGAN")
    parser.add_argument("--diffusionName", default="Phase 2: Diffusion (rna-diffusion-clean)", help="Display name for Diffusion")
    parser.add_argument("--chromatinDir", default=None, help="Optional chromatin feature pack directory")
    parser.add_argument("--featureSet", default="representative", help="Feature set name")
    parser.add_argument("--noPairFeatures", action="store_true", help="Disable pair features flag")
    args = parser.parse_args()

    compare_two_models(
        cgan_dir=args.cganDir,
        diffusion_dir=args.diffusionDir,
        npz_data_dir=args.gtDir,
        eval_out_dir=args.outDir,
        bin_size=args.binSize,
        chroms=args.chroms,
        cgan_name=args.cganName,
        diffusion_name=args.diffusionName,
        chromatin_dir=args.chromatinDir,
        feature_set=args.featureSet,
        use_pair_features=(not args.noPairFeatures),
    )


