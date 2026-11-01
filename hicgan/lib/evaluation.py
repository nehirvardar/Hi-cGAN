import os
import re
import glob
import json
import logging
import numpy as np
import pandas as pd
from scipy import sparse, stats, ndimage
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
    """Return (pearson_r, spearman_rho), safely handling constant arrays."""
    u = np.asarray(vec1, dtype=np.float64).ravel()
    v = np.asarray(vec2, dtype=np.float64).ravel()
    if u.size < 2 or np.std(u) <= 1e-12 or np.std(v) <= 1e-12:
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
        f1_vals = np.where(denom > 0, 2.0 * prec * rec / denom, 0.0)
        best_f1 = float(np.max(f1_vals)) if f1_vals.size > 0 else float("nan")
        return auroc, auprc, best_f1
    except Exception:
        return float("nan"), float("nan"), float("nan")


def parse_training_losses(train_out_dir: str) -> pd.DataFrame:
    """
    Parse epoch-by-epoch training and validation losses from `lossValues_*.npz`
    and/or `training.log` inside `train_out_dir`.
    Returns a DataFrame with columns:
      epoch, train_gen_total, val_gen_total, train_pixel, val_pixel,
      train_adv, val_adv, train_tv, val_tv,
      train_disc_total, val_disc_total, train_disc_true, train_disc_fake
    """
    rows = []
    log_path = os.path.join(train_out_dir, "training.log")
    if os.path.exists(log_path):
        epoch_pattern = re.compile(
            r"\[EPOCH-LOSS\]\s*epoch=(\d+)/(\d+)\s*\|\s*"
            r"train_gen_total=([0-9.eE+-]+),\s*train_pixel=([0-9.eE+-]+),\s*train_adv=([0-9.eE+-]+),\s*train_tv=([0-9.eE+-]+),\s*"
            r"train_disc_total=([0-9.eE+-]+),\s*train_disc_true=([0-9.eE+-]+),\s*train_disc_fake=([0-9.eE+-]+)\s*\|\s*"
            r"val_gen_total=([0-9.eE+-]+),\s*val_pixel=([0-9.eE+-]+),\s*val_adv=([0-9.eE+-]+),\s*val_tv=([0-9.eE+-]+),\s*val_disc_total=([0-9.eE+-]+)"
        )
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = epoch_pattern.search(line)
                if m:
                    rows.append({
                        "epoch": int(m.group(1)),
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

    if "genLossTrainEpochs" in keys and "genLossValEpochs" in keys:
        n_ep = len(data["genLossTrainEpochs"])
        df = pd.DataFrame({
            "epoch": np.arange(1, n_ep + 1, dtype=int),
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
    Plot epoch-wise Train vs Validation curves for Generator Total, Pixel L1/L2,
    Adversarial, TV, and Discriminator losses, marking the best validation epoch.
    """
    if df_loss is None or df_loss.empty:
        print("[EVAL-LOSS] Kaydedilmiş eğitim kayıp (loss) verisi bulunamadı.")
        return None

    os.makedirs(eval_out_dir, exist_ok=True)
    csv_path = os.path.join(eval_out_dir, "training_loss_history.csv")
    df_loss.to_csv(csv_path, index=False)

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

    epochs = df_loss["epoch"].values

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
    plt.show()
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
) -> dict:
    """
    Compute comprehensive numerical metrics comparing ground-truth and predicted
    matrices in both raw and log1p spaces for a single chromosome, including:
      - Core 5 metrics (AUPRC, Distance-Stratified Corr, Active GT>0 Corr, 1D Marginal Corr, SSIM/Cosine)
      - 0-100 Composite Quality Score
      - P(d) Distance-Decay Baseline comparison and Lift/Delta metrics
    """
    gt_raw = np.maximum(np.asarray(gt_dense_raw, dtype=np.float64), 0.0)
    pred_raw = np.maximum(np.asarray(pred_dense_raw, dtype=np.float64), 0.0)

    gt_log = np.log1p(gt_raw)
    pred_log = np.log1p(pred_raw)

    n = gt_raw.shape[0]
    triu_r, triu_c = np.triu_indices(n, k=0)
    gt_triu_raw = gt_raw[triu_r, triu_c]
    pred_triu_raw = pred_raw[triu_r, triu_c]
    gt_triu_log = gt_log[triu_r, triu_c]
    pred_triu_log = pred_log[triu_r, triu_c]

    # Error metrics (all upper-triangle pairs)
    mae_raw = float(np.mean(np.abs(gt_triu_raw - pred_triu_raw)))
    mse_raw = float(np.mean((gt_triu_raw - pred_triu_raw) ** 2))
    rmse_raw = float(np.sqrt(mse_raw))

    mae_log = float(np.mean(np.abs(gt_triu_log - pred_triu_log)))
    mse_log = float(np.mean((gt_triu_log - pred_triu_log) ** 2))
    rmse_log = float(np.sqrt(mse_log))

    # Correlations on all upper-triangle bin pairs
    pearson_all_raw, spearman_all_raw = safe_corr(gt_triu_raw, pred_triu_raw)
    pearson_all_log, spearman_all_log = safe_corr(gt_triu_log, pred_triu_log)

    # Correlations on active bin pairs (GT > 0)
    pos_mask = gt_triu_raw > 0
    n_pos = int(np.sum(pos_mask))
    prevalence_pos = float(n_pos) / float(max(gt_triu_raw.size, 1))
    if n_pos >= 2:
        pearson_pos_raw, spearman_pos_raw = safe_corr(gt_triu_raw[pos_mask], pred_triu_raw[pos_mask])
        pearson_pos_log, spearman_pos_log = safe_corr(gt_triu_log[pos_mask], pred_triu_log[pos_mask])
    else:
        pearson_pos_raw = spearman_pos_raw = pearson_pos_log = spearman_pos_log = float("nan")

    # Distance-stratified correlations (across genomic distances d >= 1 where GT has variance)
    curves = compute_distance_curves(gt_raw, pred_raw)
    dist_strat_pearson_log = float(curves["mean_dist_pearson_log"])
    dist_strat_spearman_log = float(curves["mean_dist_spearman_log"])

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
    cos_raw = compute_cosine_similarity(gt_triu_raw, pred_triu_raw)

    ssim_log = compute_ssim_2d(gt_log, pred_log)
    psnr_log = compute_psnr(gt_log, pred_log)
    cos_log = compute_cosine_similarity(gt_triu_log, pred_triu_log)

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
        "gt_nnz_triu": n_pos,
        "Prevalence_GTpos": prevalence_pos,
        "Composite_Score_100": composite_score,
        "AUPRC_GTpos": auprc_pos,
        "AUPRC_Top10pct": auprc_top10,
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
        base_metrics = evaluate_chromosome_pair(gt_raw, base_raw, chrom=chrom, include_baseline=False)
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


def compute_distance_curves(gt_dense_raw: np.ndarray, pred_dense_raw: np.ndarray):
    """
    Compute distance-dependent mean contact intensity P(d) and distance-stratified
    Pearson/Spearman correlations as a function of genomic bin distance d = |i - j|.
    For distances d >= 1 where GT has non-zero variance (std > 1e-12) and >= 5 bins,
    if prediction has zero variance, correlation at distance d is recorded as 0.0.
    """
    gt_raw = np.maximum(np.asarray(gt_dense_raw, dtype=np.float64), 0.0)
    pred_raw = np.maximum(np.asarray(pred_dense_raw, dtype=np.float64), 0.0)
    gt_log = np.log1p(gt_raw)
    pred_log = np.log1p(pred_raw)

    n = gt_raw.shape[0]
    distances = np.arange(0, n, dtype=int)
    gt_pd_raw = np.zeros(n, dtype=np.float64)
    pred_pd_raw = np.zeros(n, dtype=np.float64)
    gt_pd_log = np.zeros(n, dtype=np.float64)
    pred_pd_log = np.zeros(n, dtype=np.float64)
    pearson_by_d = np.full(n, np.nan, dtype=np.float64)
    spearman_by_d = np.full(n, np.nan, dtype=np.float64)

    active_pearson_list = []
    active_spearman_list = []

    for d in distances:
        g_diag_raw = np.diagonal(gt_raw, offset=d)
        p_diag_raw = np.diagonal(pred_raw, offset=d)
        g_diag_log = np.diagonal(gt_log, offset=d)
        p_diag_log = np.diagonal(pred_log, offset=d)

        gt_pd_raw[d] = float(np.mean(g_diag_raw))
        pred_pd_raw[d] = float(np.mean(p_diag_raw))
        gt_pd_log[d] = float(np.mean(g_diag_log))
        pred_pd_log[d] = float(np.mean(p_diag_log))

        if g_diag_log.size >= 5 and np.std(g_diag_log) > 1e-12:
            if np.std(p_diag_log) <= 1e-12:
                pr, sr = 0.0, 0.0
            else:
                pr, sr = safe_corr(g_diag_log, p_diag_log)
                pr = 0.0 if np.isnan(pr) else float(pr)
                sr = 0.0 if np.isnan(sr) else float(sr)
            pearson_by_d[d] = pr
            spearman_by_d[d] = sr
            if d >= 1:
                active_pearson_list.append(pr)
                active_spearman_list.append(sr)

    mean_dist_pearson = float(np.mean(active_pearson_list)) if active_pearson_list else 0.0
    mean_dist_spearman = float(np.mean(active_spearman_list)) if active_spearman_list else 0.0

    return {
        "distances": distances,
        "gt_pd_raw": gt_pd_raw,
        "pred_pd_raw": pred_pd_raw,
        "gt_pd_log": gt_pd_log,
        "pred_pd_log": pred_pd_log,
        "pearson_by_d": pearson_by_d,
        "spearman_by_d": spearman_by_d,
        "mean_dist_pearson_log": mean_dist_pearson,
        "mean_dist_spearman_log": mean_dist_spearman,
    }


def plot_triple_heatmaps(chrom_data_dict: dict, eval_out_dir: str, run_name: str = ""):
    """
    Plot 3-panel heatmaps for each chromosome:
      1. Ground Truth (log1p)
      2. Predicted (log1p) [shared colorbar range with GT]
      3. Residual / Difference Map (Pred_log1p - GT_log1p, symmetric 'bwr' colormap)
    """
    os.makedirs(eval_out_dir, exist_ok=True)
    saved_paths = []
    for chrom, item in chrom_data_dict.items():
        gt_log = np.log1p(item["gt_raw"])
        pred_log = np.log1p(item["pred_raw"])
        residual = pred_log - gt_log

        vmax = float(max(np.max(gt_log), np.max(pred_log), 1e-6))
        rmax = float(max(np.max(np.abs(residual)), 1e-6))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

        im0 = axes[0].imshow(gt_log, cmap="Reds", vmin=0.0, vmax=vmax)
        axes[0].set_title(f"Gerçek (Ground Truth) chr{chrom} [log1p]")
        axes[0].set_xlabel("Genomik Bin (j)")
        axes[0].set_ylabel("Genomik Bin (i)")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(pred_log, cmap="Reds", vmin=0.0, vmax=vmax)
        axes[1].set_title(f"Tahmin Edilen (Predicted) chr{chrom} [log1p]")
        axes[1].set_xlabel("Genomik Bin (j)")
        axes[1].set_ylabel("Genomik Bin (i)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(residual, cmap="bwr", vmin=-rmax, vmax=rmax)
        mae_l = float(np.mean(np.abs(residual)))
        axes[2].set_title(f"Fark / Residual (Pred - GT) chr{chrom} [MAE={mae_l:.4f}]")
        axes[2].set_xlabel("Genomik Bin (j)")
        axes[2].set_ylabel("Genomik Bin (i)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle(f"chr{chrom} — 3'lü Karşılaştırma ve Residual Haritası ({run_name})", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = os.path.join(eval_out_dir, f"heatmap_triple_chr{chrom}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.show()
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
    plt.show()

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
    plt.show()

    return [pd_path, corr_path]


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
    plt.show()
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
    plt.show()
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
    Build a concise, high-signal 'Core 5 Metrics + P(d) Baseline & Composite Score (0-100)'
    scorecard table from `df_metrics_full`.
    """
    if df_metrics_full is None or df_metrics_full.empty:
        return pd.DataFrame()

    rows = []
    for _, r in df_metrics_full.iterrows():
        rows.append({
            "Kromozom": r.get("chrom"),
            "Bileşik Skor (0-100)": float(r.get("Composite_Score_100", np.nan)),
            "Baz P(d) Skor (0-100)": float(r.get("Baseline_Composite_Score_100", np.nan)),
            "Skor Farkı (Δ vs Baz)": float(r.get("Composite_Delta_vs_Base", np.nan)),
            "1) AUPRC (GT>0)": float(r.get("AUPRC_GTpos", np.nan)),
            "Baz AUPRC (GT>0)": float(r.get("Baseline_AUPRC_GTpos", np.nan)),
            "AUPRC Katı (vs Rastgele)": float(r.get("AUPRC_Lift_vs_Random", np.nan)),
            "1b) AUPRC (Top %10)": float(r.get("AUPRC_Top10pct", np.nan)),
            "2) Katmanlı Spearman ρ(d)": float(r.get("DistStrat_Spearman_log1p", np.nan)),
            "2b) Katmanlı Pearson r(d)": float(r.get("DistStrat_Pearson_log1p", np.nan)),
            "3) Aktif Hücre r (GT>0)": float(r.get("Pearson_GTpos_log1p", np.nan)),
            "4) 1B Marjinal Profil r": float(r.get("Marginal1D_Pearson_log1p", np.nan)),
            "5) Yapısal SSIM (log1p)": float(r.get("SSIM_log1p", np.nan)),
            "5b) Kosinüs Benzerliği": float(r.get("Cosine_log1p", np.nan)),
            "Model Tanı / Yorum": _diagnose_score_row(r),
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
    plt.show()
    return out_path


def compare_runs_leaderboard(
    search_root_dirs,
    npz_data_dir: str,
    bin_size: int = 250000,
    chroms=None,
    eval_out_dir: str = None,
    max_runs: int = 8,
) -> pd.DataFrame:
    """
    Scan run directories in `search_root_dirs` (e.g. ['output_models', 'notebook_pipeline_output']),
    compute or load Core 5 Metrics + Composite Score (0-100) for each run that has predicted `.npz`
    chromosome matrices, rank them in a Leaderboard DataFrame, and plot a Multi-Run Radar & Bar chart.
    """
    from . import utils

    if isinstance(search_root_dirs, str):
        search_root_dirs = [search_root_dirs]

    chrom_filter = {str(c).lstrip("chr") for c in chroms} if chroms else None

    run_dirs = []
    seen_real = set()
    for root in search_root_dirs:
        if not root or not os.path.exists(root):
            continue
        if os.path.isdir(os.path.join(root, "predict_run")) or os.path.isdir(os.path.join(root, "train_run")):
            candidates = [root]
        else:
            candidates = sorted(glob.glob(os.path.join(root, "run_*")))
        for cdir in candidates:
            rp = os.path.realpath(cdir)
            if rp not in seen_real and os.path.isdir(cdir):
                seen_real.add(rp)
                run_dirs.append(cdir)

    leaderboard_rows = []
    for rdir in run_dirs:
        rname = os.path.basename(os.path.normpath(rdir))
        pred_npz_dir = os.path.join(rdir, "predict_run", "npz_matrices")
        train_dir = os.path.join(rdir, "train_run")
        npz_files = sorted(glob.glob(os.path.join(pred_npz_dir, "chr*.npz")))
        if not npz_files:
            continue

        # Parse run metadata from directory name or hicgan_metadata.pkl
        m_name = re.match(r"run_\d+_\d+_([^_]+)_([^_]+)_ws(\d+)_ep(\d+)", rname)
        input_mode = m_name.group(1) if m_name else "?"
        window_mode = m_name.group(2) if m_name else "?"
        ws = int(m_name.group(3)) if m_name else 64
        ep = int(m_name.group(4)) if m_name else 0

        # Check best val loss from training history if available
        df_loss = parse_training_losses(train_dir)
        best_val_loss = float(df_loss["val_gen_total"].min()) if (not df_loss.empty and "val_gen_total" in df_loss) else np.nan

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
                    crow = evaluate_chromosome_pair(gt_sp.toarray(), pr_sp.toarray(), chrom=chrom, include_baseline=True)
                    chrom_rows.append(crow)
                    chrom_names.append(f"chr{chrom}")
            except Exception as exc:
                log.debug("Skipping %s in %s: %s", pfile, rname, str(exc))

        if not chrom_rows:
            continue

        dfc = pd.DataFrame(chrom_rows)
        leaderboard_rows.append({
            "Run Adı": rname,
            "Girdi Modu": input_mode,
            "Pencere Modu": window_mode,
            "WS": ws,
            "Epoch": ep,
            "Kromozomlar": ",".join(chrom_names),
            "Bileşik Skor (0-100)": float(dfc["Composite_Score_100"].mean()),
            "Baz Skor (0-100)": float(dfc["Baseline_Composite_Score_100"].mean()),
            "Skor Farkı (Δ vs Baz)": float(dfc["Composite_Delta_vs_Base"].mean()),
            "AUPRC (GT>0)": float(dfc["AUPRC_GTpos"].mean()),
            "AUPRC (Top %10)": float(dfc["AUPRC_Top10pct"].mean()),
            "Katmanlı Spearman ρ(d)": float(dfc["DistStrat_Spearman_log1p"].mean()),
            "Aktif Hücre r (GT>0)": float(dfc["Pearson_GTpos_log1p"].mean()),
            "1B Marjinal Profil r": float(dfc["Marginal1D_Pearson_log1p"].mean()),
            "SSIM (log1p)": float(dfc["SSIM_log1p"].mean()),
            "Min Val Loss": best_val_loss,
        })

    if not leaderboard_rows:
        return pd.DataFrame()

    df_lb = pd.DataFrame(leaderboard_rows).sort_values("Bileşik Skor (0-100)", ascending=False).reset_index(drop=True)
    df_lb.insert(0, "Sıra", np.arange(1, len(df_lb) + 1))

    if eval_out_dir:
        os.makedirs(eval_out_dir, exist_ok=True)
        lb_csv = os.path.join(eval_out_dir, "multi_run_leaderboard.csv")
        df_lb.to_csv(lb_csv, index=False)

        # Plot Multi-Run Radar & Leaderboard Bar Chart (top `max_runs` runs)
        df_plot = df_lb.head(max_runs)
        labels = [
            "AUPRC\n(GT>0)",
            "AUPRC\n(Top %10)",
            "Katmanlı\nρ(d)",
            "Aktif Hücre\nr (GT>0)",
            "1B Marjinal\nProfil r",
            "SSIM\n(log1p)",
        ]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles_closed = angles + [angles[0]]

        fig = plt.figure(figsize=(16, 6.2))
        ax_radar = fig.add_subplot(1, 2, 1, polar=True)
        ax_bar = fig.add_subplot(1, 2, 2)
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(df_plot), 2)))

        for idx, (_, r) in enumerate(df_plot.iterrows()):
            vals = [
                float(np.clip(r["AUPRC (GT>0)"], 0.0, 1.0)),
                float(np.clip(r["AUPRC (Top %10)"], 0.0, 1.0)),
                float(np.clip(r["Katmanlı Spearman ρ(d)"], 0.0, 1.0)),
                float(np.clip(r["Aktif Hücre r (GT>0)"], 0.0, 1.0)),
                float(np.clip(r["1B Marjinal Profil r"], 0.0, 1.0)),
                float(np.clip(r["SSIM (log1p)"], 0.0, 1.0)),
            ]
            vals_closed = vals + [vals[0]]
            short_name = str(r["Run Adı"]).replace("run_", "")
            c = cmap[idx % len(cmap)]
            ax_radar.plot(angles_closed, vals_closed, "o-", lw=2, color=c, label=f"#{int(r['Sıra'])} {short_name}")
            ax_radar.fill(angles_closed, vals_closed, color=c, alpha=0.08)

        ax_radar.set_thetagrids(np.degrees(angles), labels, fontsize=9.5)
        ax_radar.set_ylim(0.0, 1.0)
        ax_radar.set_title("Çoklu Run Radar Karşılaştırması", pad=18, fontsize=12, fontweight="bold")
        ax_radar.legend(loc="upper right", bbox_to_anchor=(1.32, 1.15), fontsize=8.5)

        y_pos = np.arange(len(df_plot))
        short_names = [f"#{int(r['Sıra'])} {str(r['Run Adı']).replace('run_', '')}" for _, r in df_plot.iterrows()]
        scores = df_plot["Bileşik Skor (0-100)"].values
        bases = df_plot["Baz Skor (0-100)"].values

        ax_bar.barh(y_pos, scores, height=0.55, color=[cmap[i % len(cmap)] for i in range(len(df_plot))], alpha=0.85, label="Run Bileşik Skoru")
        ax_bar.axvline(float(np.mean(bases)), color="#333333", ls="--", lw=1.5, label=f"P(d) Baz Skor Ort. ({np.mean(bases):.1f})")
        for i, sc in enumerate(scores):
            ax_bar.text(sc + 0.8, i, f"{sc:.2f}/100", va="center", ha="left", fontsize=9.5, fontweight="bold")

        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(short_names, fontsize=9)
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, max(max(scores), float(np.mean(bases)), 50.0) * 1.25)
        ax_bar.set_xlabel("Bileşik Model Kalite Skoru (0 - 100)")
        ax_bar.set_title("Run Liderlik Sıralaması (Composite Score)", fontsize=12, fontweight="bold")
        ax_bar.grid(True, axis="x", alpha=0.3)
        ax_bar.legend(loc="lower right")

        fig.suptitle("Çoklu Run (Deney) Performans Karşılaştırması ve Liderlik Tablosu", fontsize=13.5, fontweight="bold")
        plt.tight_layout()
        radar_path = os.path.join(eval_out_dir, "multi_run_comparison_radar.png")
        fig.savefig(radar_path, dpi=300, bbox_inches="tight")
        plt.show()

    return df_lb

