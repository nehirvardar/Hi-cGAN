import argparse
import os
import numpy as np
import scipy.sparse as sp
import cv2
import cooler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from lib import hicGAN


def evaluate_chromosome(
    model_path,
    chrom,
    rna_path,
    gt_path,
    window_size=128,
    sample_steps=50,
    num_windows=5,
    output_dir="test_results"
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Hi-C ground truth for {chrom} from {gt_path}...")
    c = cooler.Cooler(gt_path)
    fetch_chrom = chrom if chrom in c.chromnames else chrom.replace("chr", "")
    if fetch_chrom not in c.chromnames:
        fetch_chrom = "chr" + chrom
    if fetch_chrom not in c.chromnames:
        fetch_chrom = c.chromnames[0]
    gt_matrix = c.matrix(balance=False, sparse=True).fetch(fetch_chrom).tocsr()
    target_dim = gt_matrix.shape[0]

    print(f"Loading RNA data for {chrom} from {rna_path}...")
    rna_sparse = sp.load_npz(rna_path)
    rna_dense = rna_sparse.toarray().astype(np.float32)

    print(f"Resizing RNA from {rna_dense.shape} to target {target_dim}x{target_dim}...")
    rna_resized = cv2.resize(rna_dense, (target_dim, target_dim), interpolation=cv2.INTER_LINEAR)

    print(f"Initializing HiCGAN diffusion model (window_size={window_size})...")
    model = hicGAN.HiCGAN(log_dir=output_dir, number_factors=1, input_size=window_size)
    model.loaddenoiser(model_path)

    flanking = window_size
    max_idx = target_dim - (window_size + 2 * flanking)

    step_stride = max(1, max_idx // 1000)
    scores = []
    for idx in range(0, max_idx, step_stride):
        start = idx + flanking
        stop = start + window_size
        rna_sub = rna_resized[start:stop, start:stop]
        rna_sum = np.sum(np.abs(rna_sub))
        if rna_sum > 0:
            scores.append((rna_sum, idx))

    scores.sort(reverse=True, key=lambda x: x[0])
    selected_indices = [idx for _, idx in scores[:num_windows]]

    if len(selected_indices) == 0:
        selected_indices = [max_idx // 2 + i * window_size for i in range(num_windows)]

    print(f"Evaluating {len(selected_indices)} windows {sample_steps} diffusion steps")

    correlations = []
    mses = []

    for i, idx in enumerate(selected_indices):
        start = idx + flanking
        stop = start + window_size

        rna_win = rna_resized[start:stop, start:stop]
        gt_win = np.array(np.nan_to_num(gt_matrix[start:stop, start:stop].todense()), dtype=np.float32)

        gt_max = gt_win.max()
        if gt_max > 0:
            gt_norm = gt_win / gt_max
        else:
            gt_norm = gt_win

        rna_tensor = tf.convert_to_tensor(rna_win[np.newaxis, :, :, np.newaxis], dtype=tf.float32)

        #Denoising
        pred_tensor = model.predictionStep(rna_tensor, training=False, sample_steps=sample_steps)
        pred_win = np.squeeze(pred_tensor.numpy())

        #Metrics
        mse = float(np.mean((gt_norm - pred_win) ** 2))
        mask = (gt_norm > 0) | (pred_win > 0)
        if mask.sum() > 10:
            corr = float(np.corrcoef(gt_norm[mask].ravel(), pred_win[mask].ravel())[0, 1])
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        correlations.append(corr)
        mses.append(mse)

        # Görselleştirme
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        im0 = axs[0].imshow(rna_win, cmap="coolwarm")
        axs[0].set_title(f"RNA Input (TSS)\nBins {start}-{stop}")
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        im1 = axs[1].imshow(gt_norm, cmap="coolwarm")
        axs[1].set_title("Ground Truth Hi-C")
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        im2 = axs[2].imshow(pred_win, cmap="coolwarm")
        axs[2].set_title(f"Diffusion Predicted Hi-C\nMSE: {mse:.4f} | Pearson: {corr:.4f}")
        fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

        fig.tight_layout()
        figname = os.path.join(output_dir, f"test_{chrom}_win_{i+1}_bin_{start}.png")
        fig.savefig(figname, dpi=150)
        plt.close(fig)
        print(f"Pencere {i+1} (bin {start}): MSE = {mse:.4f}, Pearson r = {corr:.4f} -> Kaydedildi: {figname}")

    print("\n" + "=" * 50)
    print(f"Evaluation ({chrom}):")
    print(f"Average MSE: {np.mean(mses):.4f}")
    print(f"Average Pearson Correlation: {np.mean(correlations):.4f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Hi-C Diffusion Model Testing and Evaluating ")
    parser.add_argument("--model", "-m", required=True, help="directory of ckpt")
    parser.add_argument("--chrom", "-c", default="chr1", help="Test chromosome")
    parser.add_argument("--rna", default="", help="RNA .npz file directory (default: data/rna/{chrom}.npz)")
    parser.add_argument("--gt", default="", help="GT Hi-C .cool file directory (default: data/gt/{chrom}.cool)")
    parser.add_argument("--windowSize", "-ws", type=int, default=128, help="Window size (default: 128)")
    parser.add_argument("--steps", "-s", type=int, default=1000, help="Step count (default: 1000)")
    parser.add_argument("--numWindows", "-n", type=int, default=5, help="TTest window count (default: 5)")
    parser.add_argument("--outputDir", "-o", default="test_results", help="output directory")

    args = parser.parse_args()

    chrom = args.chrom if args.chrom.startswith("chr") else "chr" + args.chrom
    rna_path = args.rna or os.path.join("data", "rna", f"{chrom}.npz")
    gt_path = args.gt or os.path.join("data", "gt", f"{chrom}.cool")

    evaluate_chromosome(
        model_path=args.model,
        chrom=chrom,
        rna_path=rna_path,
        gt_path=gt_path,
        window_size=args.windowSize,
        sample_steps=args.steps,
        num_windows=args.numWindows,
        output_dir=args.outputDir
    )


if __name__ == "__main__":
    main()
