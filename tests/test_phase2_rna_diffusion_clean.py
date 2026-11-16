import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest

import numpy as np
from scipy import sparse

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hicgan.lib import dataContainer, evaluation, featurePack, hicGAN, utils
from hicgan import predict as hic_predict
from hicgan import training as hic_training


def _make_synthetic_chrom_npz(out_dir: str, chrom: str, n_bins: int = 80, bin_size: int = 250000, seed: int = 123):
    rng = np.random.default_rng(seed)
    mat = np.zeros((n_bins, n_bins), dtype=np.float32)
    for i in range(n_bins):
        for j in range(i, min(n_bins, i + 20)):
            d = j - i
            base_rate = max(0.05, 8.0 * np.exp(-0.22 * d))
            if rng.random() < (0.75 if d <= 6 else 0.25):
                val = float(rng.poisson(base_rate))
                mat[i, j] = val
                mat[j, i] = val
    os.makedirs(out_dir, exist_ok=True)
    sp = sparse.csr_matrix(mat)
    npz_path = os.path.join(out_dir, f"chr{chrom}.npz")
    sparse.save_npz(npz_path, sp)
    return mat, npz_path


def _make_synthetic_feature_pack(out_dir: str, chroms, n_bins: int = 80, seed: int = 123):
    rng = np.random.default_rng(seed)
    cache_dir = os.path.join(out_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)

    feat_names = [f"feat_{j:02d}" for j in range(10)] + ["ind_has_pc_gene", "ind_has_mirna_expr", "ind_has_hela_data"]
    feat_sheets = ["direct"] * 10 + ["indicator"] * 3
    feat_tiers = ["1"] * 10 + ["0"] * 3
    feat_rep = ["yes"] * 13
    feat_cell = ["GM12878"] * 10 + ["none", "GM12878", "HeLa"]
    n_feats = len(feat_names)

    # Write feature_tiers.csv so featurePack.is_feature_pack_source(out_dir) returns True
    with open(os.path.join(out_dir, "feature_tiers.csv"), "w", encoding="utf-8") as fh:
        fh.write("feature_name,sheet,tier,representative,cell_line\n")
        for nm, sh, tr, rp, cl in zip(feat_names, feat_sheets, feat_tiers, feat_rep, feat_cell):
            fh.write(f"{nm},{sh},{tr},{rp},{cl}\n")

    all_chroms = []
    all_bin_idx = []
    all_feats = []
    for c in chroms:
        all_chroms.extend([str(c)] * n_bins)
        all_bin_idx.append(np.arange(n_bins, dtype=np.int32))
        mat_f = np.clip(rng.uniform(0.0, 1.0, size=(n_bins, n_feats)), 0.0, 1.0).astype(np.float32)
        all_feats.append(mat_f)

        pair_mat = np.zeros((n_bins, n_bins), dtype=np.float32)
        for i in range(n_bins):
            for j in range(i, min(n_bins, i + 12)):
                val = float(np.exp(-0.15 * (j - i)))
                pair_mat[i, j] = val
                pair_mat[j, i] = val
        sparse.save_npz(os.path.join(cache_dir, f"pair_kmer7_chr{c}.npz"), sparse.csr_matrix(pair_mat))

    np.savez_compressed(
        os.path.join(cache_dir, "bin_features_cache.npz"),
        chroms=np.asarray(all_chroms, dtype="<U8"),
        matrix_bin_index=np.concatenate(all_bin_idx),
        feature_names=np.asarray(feat_names, dtype="<U64"),
        feature_sheets=np.asarray(feat_sheets, dtype="<U32"),
        feature_tiers=np.asarray(feat_tiers, dtype="<U8"),
        feature_rep=np.asarray(feat_rep, dtype="<U8"),
        feature_cell_line=np.asarray(feat_cell, dtype="<U32"),
        features_norm=np.vstack(all_feats).astype(np.float32),
    )
    return out_dir


class TestPhase2RnaDiffusionClean(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_root = tempfile.mkdtemp(prefix="phase2_diff_test_")
        cls.gt_dir = os.path.join(cls.tmp_root, "gt_npz")
        cls.fp_dir = os.path.join(cls.tmp_root, "feature_pack")
        cls.chrom_sizes_path = os.path.join(cls.tmp_root, "chrom.sizes")

        cls.chroms = ["1", "20", "21"]
        cls.n_bins = 80
        cls.bin_size = 250000

        idx_rows = []
        offset = 0
        with open(cls.chrom_sizes_path, "w", encoding="utf-8") as fh:
            for idx, c in enumerate(cls.chroms):
                mat, _ = _make_synthetic_chrom_npz(cls.gt_dir, c, n_bins=cls.n_bins, bin_size=cls.bin_size, seed=100 + idx)
                nnz = int(np.count_nonzero(mat))
                idx_rows.append(f"{c},{cls.n_bins},{offset},{nnz}\n")
                offset += cls.n_bins
                fh.write(f"chr{c}\t{cls.n_bins * cls.bin_size}\n")
        with open(os.path.join(cls.gt_dir, "chrom_index.csv"), "w", encoding="utf-8") as fh:
            fh.write("chrom,n_bins,offset,nnz\n")
            fh.writelines(idx_rows)

        _make_synthetic_feature_pack(cls.fp_dir, cls.chroms, n_bins=cls.n_bins, seed=200)

        cls.split_path = os.path.join(cls.tmp_root, "splits.txt")
        with open(cls.split_path, "w", encoding="utf-8") as fh:
            fh.write("chr1 train\nchr20 val\nchr21 test\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_root, ignore_errors=True)

    def test_01_splits_integrity(self):
        real_split = os.path.join(REPO_ROOT, "hicgan", "data", "splits.txt")
        self.assertTrue(os.path.exists(real_split))
        parsed = utils.parseSplitFile(real_split)
        self.assertEqual(set(parsed.keys()), {"train", "val", "test"})
        tr, va, te = set(parsed["train"]), set(parsed["val"]), set(parsed["test"])
        self.assertTrue(tr.isdisjoint(va))
        self.assertTrue(tr.isdisjoint(te))
        self.assertTrue(va.isdisjoint(te))
        self.assertEqual(len(tr) + len(va) + len(te), 23)

    def test_02_p99_scaling_train_only_isolation(self):
        norm_stats = utils.computeGlobalP99TargetRange(
            pMatrixFilePaths=[self.gt_dir],
            pTrainChromosomes=["1"],
            scale_factor=10.0,
            clip_percentile=99.0,
        )
        self.assertEqual(norm_stats["method"], "p99_log10")
        self.assertEqual(norm_stats["train_chromosomes"], ["1"])
        self.assertAlmostEqual(norm_stats["min"], 0.0)
        self.assertGreater(norm_stats["hi"], 0.0)

        x = np.array([[[[0.0], [1.0], [10.0], [1000.0]]]], dtype=np.float32)
        u = utils.toP99LogUnitRange(x, norm_stats["hi"], scale_factor=10.0)
        self.assertGreaterEqual(float(np.min(u)), 0.0)
        self.assertLessEqual(float(np.max(u)), 1.0)
        rec = utils.fromP99LogUnitRange(u, norm_stats["hi"], scale_factor=10.0)
        self.assertAlmostEqual(float(rec[0, 0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(rec[0, 0, 1, 0]), 1.0, places=4)

    def test_03_datacontainer_feature_pack_sat_filter_and_mirror(self):
        dc = dataContainer.DataContainer(
            chromosome="1",
            matrixFilePath=self.gt_dir,
            chromatinFolder=self.fp_dir,
            binSize=self.bin_size,
            inputMode="feature_pack",
            flankPadChromosomes=["all"],
            featureSet="representative",
            usePairFeatures=True,
            filterEmptyWindows=True,
        )
        dc.loadData(
            windowSize=64,
            flankingSize=None,
            maximumDistance=0,
            scaleFeatures=True,
            clampFeatures=False,
            scaleTargets=False,
            windowMode="grid",
            gridStride=16,
            flankPadChromosomes=["all"],
            featureSet="representative",
            usePairFeatures=True,
            filterEmptyWindows=True,
        )
        self.assertGreater(dc.getNumberSamples(), 0)
        sample = dc.getSampleData(0)
        f_shape = sample["factorData"].shape
        t_shape = sample["out_matrixData"].shape
        self.assertEqual(t_shape, (64, 64, 1))
        # (192, 13*2 + 2 + 64, 1) -> shape==(192, 92, 1)
        self.assertEqual(f_shape, (192, 92, 1))

        mir_inp, mir_tgt = hic_training.records.mirror_function(
            tf.convert_to_tensor(sample["factorData"]),
            tf.convert_to_tensor(sample["out_matrixData"]),
            is_2d_input=False,
            is_grid_mode=True,
            has_pair_2d=True,
        )
        expected_tgt = np.flip(np.flip(sample["out_matrixData"], axis=0), axis=1)
        np.testing.assert_allclose(mir_tgt["out_matrixData"].numpy(), expected_tgt, atol=1e-6)
        self.assertEqual(tuple(mir_inp["factorData"].shape), f_shape)

    def test_04_diffusion_architecture_no_discriminator_and_loss(self):
        out_dir = os.path.join(self.tmp_root, "arch_test")
        os.makedirs(out_dir, exist_ok=True)
        n_factors = 92  # 13*2 1D + 2 pos + 64 pair
        model = hicGAN.HiCGAN(
            log_dir=out_dir,
            number_factors=n_factors,
            timesteps=40,
            sample_steps=8,
            dropout_rate=0.0,
            input_size=64,
            use_embedding=True,
            enforce_symmetry=False,
            use_pair_features=True,
        )

        # Phase 2 must have NO discriminator
        self.assertFalse(hasattr(model, "discriminator"))
        self.assertFalse(hasattr(model, "discriminator_optimizer"))
        self.assertTrue(hasattr(model, "denoiser"))
        self.assertTrue(hasattr(model, "unet_denoiser"))
        self.assertTrue(hasattr(model, "condition_encoder"))

        # Check SpatialTimeEncoding shape and uniform spatial broadcast
        ste = hicGAN.SpatialTimeEncoding(timesteps=40)
        t_idx = tf.constant([5, 19], dtype=tf.int32)
        noisy = tf.random.normal((2, 64, 64, 1), seed=8)
        t_map = ste([noisy, t_idx])
        self.assertEqual(tuple(t_map.shape), (2, 64, 64, 1))
        self.assertAlmostEqual(float(t_map[0, 0, 0, 0]), 5.0 / 40.0, places=5)
        self.assertAlmostEqual(float(t_map[0, 31, 45, 0]), 5.0 / 40.0, places=5)
        self.assertAlmostEqual(float(t_map[1, 0, 0, 0]), 19.0 / 40.0, places=5)

        # Condition encoder shape parity: (B, 64, 64, 16)
        cond_inp_4d = tf.random.normal((2, 192, n_factors, 1), seed=7)
        cond_inp_3d = tf.squeeze(cond_inp_4d, axis=-1)
        x_emb = model.condition_encoder(cond_inp_3d, training=False)
        self.assertEqual(tuple(x_emb.shape), (2, 64, 64, 16))

        # Denoiser output is unconstrained linear (not clamped to [0, 1])
        eps_hat = model.denoiser([noisy, cond_inp_3d, t_idx], training=False)
        self.assertEqual(tuple(eps_hat.shape), (2, 64, 64, 1))
        self.assertTrue(np.any(eps_hat.numpy() < 0.0) or np.any(eps_hat.numpy() > 1.0))

        # Dropout=0.0 determinism in training=True vs dropout=0.5 stochasticity
        out_det_1 = model.denoiser([noisy, cond_inp_3d, t_idx], training=True).numpy()
        out_det_2 = model.denoiser([noisy, cond_inp_3d, t_idx], training=True).numpy()
        np.testing.assert_allclose(out_det_1, out_det_2, atol=1e-6)

        model_drop = hicGAN.HiCGAN(
            log_dir=out_dir,
            number_factors=n_factors,
            timesteps=40,
            sample_steps=8,
            dropout_rate=0.5,
            input_size=64,
            use_embedding=True,
            enforce_symmetry=False,
            use_pair_features=True,
        )
        out_stoc_1 = model_drop.denoiser([noisy, cond_inp_3d, t_idx], training=True).numpy()
        out_stoc_2 = model_drop.denoiser([noisy, cond_inp_3d, t_idx], training=True).numpy()
        self.assertGreater(float(np.max(np.abs(out_stoc_1 - out_stoc_2))), 1e-4)

        # Weighted diffusion_loss checks
        tgt = np.zeros((2, 64, 64, 1), dtype=np.float32)
        tgt[:, 10:20, 10:20, :] = 0.8
        noise_true = np.zeros_like(tgt)
        alpha_bar = tf.constant([[[[0.5]]], [[[0.5]]]], dtype=tf.float32)
        noisy_zero_eps = tf.sqrt(alpha_bar) * tf.constant(tgt)

        # Error on foreground only vs error on background only
        eps_fg_err = np.zeros_like(tgt)
        eps_fg_err[:, 12:14, 12:14, :] = 0.5
        eps_bg_err = np.zeros_like(tgt)
        eps_bg_err[:, 40:42, 40:42, :] = 0.5

        tot_fg, n_fg, _, _ = model.diffusion_loss(
            tf.constant(tgt), tf.constant(noise_true), tf.constant(eps_fg_err), noisy_zero_eps, alpha_bar
        )
        tot_bg, n_bg, _, bg_pen = model.diffusion_loss(
            tf.constant(tgt), tf.constant(noise_true), tf.constant(eps_bg_err), noisy_zero_eps, alpha_bar
        )
        self.assertGreater(float(n_fg), float(n_bg) * 5.0)
        self.assertGreater(float(bg_pen), 0.0)

        # train_step produces finite losses & deterministic validationStep matches across calls
        tot_tr, n_tr, x0_tr, bg_tr = model.train_step(cond_inp_4d, tf.constant(tgt))
        for val in (tot_tr, n_tr, x0_tr, bg_tr):
            self.assertTrue(np.isfinite(float(val)))

        v1 = [float(x) for x in model.validationStep(cond_inp_4d, tf.constant(tgt))]
        v2 = [float(x) for x in model.validationStep(cond_inp_4d, tf.constant(tgt))]
        np.testing.assert_allclose(v1, v2, atol=1e-6)

        # generate_images slices center (64, 64) 2D pair window from 4D (B, 192, 92, 1) feature_pack tensor
        from unittest import mock
        with mock.patch("hicgan.lib.evaluation.render_comparison_panels") as mock_rcp:
            model.generate_images(model.denoiser, {"factorData": cond_inp_4d}, {"out_matrixData": tf.constant(tgt)}, epoch=0)
            mock_rcp.assert_called_once()
            _, rcp_kw = mock_rcp.call_args
            self.assertEqual(rcp_kw["input_mat"].shape, (64, 64))
            self.assertEqual(rcp_kw["input_title"], "Girdi (2D Pair Window)")

    def test_05_ddim_caching_determinism_and_save_load(self):
        out_dir = os.path.join(self.tmp_root, "ddim_cache_test")
        os.makedirs(out_dir, exist_ok=True)
        n_factors = 92
        model = hicGAN.HiCGAN(
            log_dir=out_dir,
            number_factors=n_factors,
            timesteps=100,
            sample_steps=10,
            dropout_rate=0.0,
            input_size=64,
            use_embedding=True,
            enforce_symmetry=False,
            use_pair_features=True,
        )
        cond_inp_4d = tf.random.normal((2, 192, n_factors, 1), seed=9)
        cond_inp_3d = tf.squeeze(cond_inp_4d, axis=-1)

        # Determinism & [0, 1] output range
        p1 = model.predictionStep(cond_inp_4d, training=False).numpy()
        p2 = model.predictionStep(cond_inp_4d, training=False).numpy()
        np.testing.assert_allclose(p1, p2, atol=1e-6)
        self.assertGreaterEqual(float(np.min(p1)), 0.0)
        self.assertLessEqual(float(np.max(p1)), 1.0)

        # Verify cached condition_encoder DDIM matches direct uncached full-denoiser DDIM
        stride = max(1, model.timesteps // 10)
        x = tf.random.stateless_normal(shape=(2, 64, 64, 1), seed=[42, 42], dtype=tf.float32)
        for t_step in range(model.timesteps, 0, -stride):
            t_b = tf.fill([2, 1], tf.cast(t_step, tf.float32))
            t_prev = max(0, t_step - stride)
            eps_hat = tf.cast(model.denoiser([x, cond_inp_3d, t_b], training=False), tf.float32)
            ab_t = model.alphas_bar[t_step]
            ab_prev = model.alphas_bar[t_prev]
            sqrt_ab_t = tf.sqrt(tf.maximum(ab_t, 1e-8))
            sqrt_one_minus_ab_t = tf.sqrt(tf.maximum(1.0 - ab_t, 1e-8))
            x0_pred = tf.clip_by_value((x - sqrt_one_minus_ab_t * eps_hat) / sqrt_ab_t, 0.0, 1.0)
            pred_eps = (x - sqrt_ab_t * x0_pred) / sqrt_one_minus_ab_t
            x = tf.sqrt(ab_prev) * x0_pred + tf.sqrt(tf.maximum(1.0 - ab_prev, 0.0)) * pred_eps
        uncached = tf.clip_by_value(x, 0.0, 1.0).numpy()
        np.testing.assert_allclose(p1, uncached, atol=1e-5)

        # Save & reload denoiser and re-verify cached prediction equivalence
        save_path = os.path.join(out_dir, "denoiser_best.keras")
        model.denoiser.save(save_path)

        reloaded = hicGAN.HiCGAN(
            log_dir=out_dir,
            number_factors=n_factors,
            timesteps=100,
            sample_steps=10,
            dropout_rate=0.0,
            input_size=64,
            use_embedding=True,
            enforce_symmetry=False,
            use_pair_features=True,
        )
        reloaded.load_denoiser(save_path)
        self.assertIsNotNone(reloaded.condition_encoder)
        self.assertIsNotNone(reloaded.unet_denoiser)
        p_reload = reloaded.predictionStep(cond_inp_4d, training=False).numpy()
        np.testing.assert_allclose(p1, p_reload, atol=1e-5)

    def test_06_blend_modes_symmetry_and_grid_parity(self):
        n = 80
        ws = 64
        stride = 16
        starts = list(range(0, n - ws + 1, stride))
        if starts[-1] + ws < n:
            starts.append(n - ws)
        self.assertEqual(starts, [0, 16])

        coords = []
        patches = []
        rng = np.random.default_rng(7)
        for r0 in starts:
            for c0 in starts:
                if r0 <= c0:
                    coords.append((r0, c0))
                    patches.append(rng.uniform(0.0, 5.0, size=(ws, ws)).astype(np.float32))
        patches = np.asarray(patches, dtype=np.float32)

        for mode in ("raised_cosine_peak", "uniform"):
            rebuilt = utils.rebuildMatrixGrid(
                pArrayOfWindows=patches,
                pWindowCoords=coords,
                pWindowSize=ws,
                pMatrixSize=n,
                blend_mode=mode,
            )
            self.assertEqual(rebuilt.shape, (n, n))
            rebuilt_dense = rebuilt.toarray() if sparse.isspmatrix(rebuilt) else np.asarray(rebuilt)
            np.testing.assert_allclose(rebuilt_dense, rebuilt_dense.T, atol=1e-6)
            self.assertGreaterEqual(float(np.min(rebuilt_dense)), 0.0)

    def test_07_e2e_training_predict_eval_and_phased_modes(self):
        # Also exercises edge case: tiny chr1 has fewer grid windows than batchSize=16
        # (with flipSamples -> 12 < 16); drop_remainder guard ensures non-NaN training!
        train_out = os.path.join(self.tmp_root, "train_e2e")
        pred_out = os.path.join(self.tmp_root, "pred_e2e")
        phased_out = os.path.join(self.tmp_root, "pred_phased")

        train_args = [
            "--dataDir", self.gt_dir,
            "--trainingChromatinFolders", self.fp_dir,
            "--validationChromatinFolders", self.fp_dir,
            "--splitFile", self.split_path,
            "--outputFolder", train_out,
            "--binSize", str(self.bin_size),
            "--windowSize", "64",
            "--windowMode", "grid",
            "--gridStride", "16",
            "--inputMode", "feature_pack",
            "--featureSet", "representative",
            "--targetNorm", "p99_log10",
            "--flankPadChromosomes", "all",
            "--epochs", "2",
            "--batchSize", "16",
            "--timesteps", "20",
            "--sampleSteps", "4",
            "--dropout", "0.0",
            "--earlyStoppingPatience", "5",
            "--plotFrequency", "0",
            "--keepTFRecords",
        ]
        hic_training.main(train_args)

        self.assertTrue(os.path.exists(os.path.join(train_out, "denoiser_best.keras")))
        self.assertTrue(os.path.exists(os.path.join(train_out, "denoiser_final.keras")))
        self.assertTrue(os.path.exists(os.path.join(train_out, "target_value_range.json")))
        self.assertTrue(os.path.exists(os.path.join(train_out, "hicgan_metadata.pkl")))
        with open(os.path.join(train_out, "hicgan_metadata.pkl"), "rb") as mf:
            meta = pickle.load(mf)
        self.assertEqual(meta.get("modelType"), "diffusion")
        self.assertEqual(meta.get("timesteps"), 20)
        self.assertEqual(meta.get("sampleSteps"), 4)

        # Verify loss parser & plotter handle Diffusion logs/npz without NaNs
        df_loss = evaluation.parse_training_losses(train_out)
        self.assertEqual(len(df_loss), 2)
        self.assertEqual(df_loss["model_type"].iloc[0], "diffusion")
        self.assertFalse(df_loss["train_total"].isna().any())
        self.assertFalse(df_loss["val_total"].isna().any())
        loss_fig = evaluation.plot_training_loss_analysis(df_loss, os.path.join(train_out, "loss_eval"), run_name="smoke_diff")
        self.assertTrue(os.path.exists(loss_fig))

        # Test --trainOnly reuse of existing TFRecords
        hic_training.main([
            "--outputFolder", train_out,
            "--epochs", "2",
            "--batchSize", "16",
            "--timesteps", "20",
            "--sampleSteps", "4",
            "--plotFrequency", "0",
            "--trainOnly",
        ])
        self.assertTrue(os.path.exists(os.path.join(train_out, "denoiser_best.keras")))

        # Run full predict + unified eval on test split (chr21)
        predict_args_common = [
            "--trainedModel", train_out,
            "--dataDir", self.gt_dir,
            "--predictionChromosomesFolders", self.fp_dir,
            "--splitFile", self.split_path,
            "--split", "test",
            "--binSize", str(self.bin_size),
            "--windowSize", "64",
            "--windowMode", "grid",
            "--gridStride", "16",
            "--inputMode", "feature_pack",
            "--featureSet", "representative",
            "--flankPadChromosomes", "all",
            "--chromSizesPath", self.gt_dir,
            "--batchSize", "4",
            "--blendMode", "raised_cosine_peak",
        ]
        hic_predict.main(predict_args_common + ["--outputFolder", pred_out, "--mode", "all", "--runEval"])

        pred_npz = os.path.join(pred_out, "npz_matrices", "chr21.npz")
        self.assertTrue(os.path.exists(pred_npz))
        pred_sp = sparse.load_npz(pred_npz)
        self.assertEqual(pred_sp.shape, (self.n_bins, self.n_bins))
        dense_p = pred_sp.toarray()
        np.testing.assert_allclose(dense_p, dense_p.T, atol=1e-6)
        self.assertTrue(os.path.exists(os.path.join(pred_out, "evaluation", "core_metrics_scorecard.csv")))
        self.assertTrue(os.path.exists(os.path.join(pred_out, "evaluation", "evaluation_metrics_full.csv")))

        # Verify phased predict (--mode create-data -> predict -> make-matrix)
        hic_predict.main(predict_args_common + ["--outputFolder", phased_out, "--mode", "create-data"])
        hic_predict.main(["--outputFolder", phased_out, "--mode", "predict", "--batchSize", "4"])
        hic_predict.main(["--outputFolder", phased_out, "--mode", "make-matrix"])
        self.assertTrue(os.path.exists(os.path.join(phased_out, "npz_matrices", "chr21.npz")))

        # Save path for test_08
        self.__class__.diff_train_dir = train_out
        self.__class__.diff_pred_dir = pred_out

    def test_08_fair_two_model_comparison_and_leaderboard(self):
        # Build a mock Phase 1 cGAN run directory matching Phase 1 artifacts
        cgan_root = os.path.join(self.tmp_root, "mock_cgan_run")
        cgan_train = os.path.join(cgan_root, "train_run")
        cgan_npz_dir = os.path.join(cgan_root, "predict_run", "npz_matrices")
        diff_root = os.path.join(self.tmp_root, "diff_run_combined")
        diff_train = os.path.join(diff_root, "train_run")
        diff_npz_dir = os.path.join(diff_root, "predict_run", "npz_matrices")
        os.makedirs(cgan_train, exist_ok=True)
        os.makedirs(cgan_npz_dir, exist_ok=True)
        os.makedirs(diff_train, exist_ok=True)
        os.makedirs(diff_npz_dir, exist_ok=True)

        # Copy Diffusion artifacts from test_07 (run test_07 if executed individually)
        if not hasattr(self.__class__, "diff_pred_dir"):
            self.test_07_e2e_training_predict_eval_and_phased_modes()

        shutil.copy2(os.path.join(self.diff_train_dir, "hicgan_metadata.pkl"), os.path.join(diff_train, "hicgan_metadata.pkl"))
        shutil.copy2(os.path.join(self.diff_train_dir, "training.log"), os.path.join(diff_train, "training.log"))
        shutil.copy2(os.path.join(self.diff_pred_dir, "predictions.pkl"), os.path.join(diff_root, "predict_run", "predictions.pkl"))
        shutil.copy2(os.path.join(self.diff_pred_dir, "npz_matrices", "chr21.npz"), os.path.join(diff_npz_dir, "chr21.npz"))

        # Create matched Phase 1 cGAN artifacts
        with open(os.path.join(diff_train, "hicgan_metadata.pkl"), "rb") as fh:
            diff_meta = pickle.load(fh)
        cgan_meta = dict(diff_meta)
        cgan_meta["modelType"] = "cgan"
        cgan_meta.pop("timesteps", None)
        cgan_meta.pop("sampleSteps", None)
        with open(os.path.join(cgan_train, "hicgan_metadata.pkl"), "wb") as fh:
            pickle.dump(cgan_meta, fh)
        with open(os.path.join(cgan_root, "predict_run", "predictions.pkl"), "wb") as fh:
            pickle.dump({"blendMode": "raised_cosine_peak", "binSize": self.bin_size, "windowSize": 64}, fh)
        with open(os.path.join(cgan_train, "training.log"), "w", encoding="utf-8") as fh:
            fh.write(
                "[EPOCH-LOSS] epoch=1/2 | train_gen_total=12.5, train_pixel=0.120, train_adv=0.69, train_tv=0.001, "
                "train_disc_total=1.38, train_disc_true=0.69, train_disc_fake=0.69 | "
                "val_gen_total=11.8, val_pixel=0.115, val_adv=0.68, val_tv=0.001, val_disc_total=1.37\n"
                "[EPOCH-LOSS] epoch=2/2 | train_gen_total=10.5, train_pixel=0.100, train_adv=0.69, train_tv=0.001, "
                "train_disc_total=1.39, train_disc_true=0.69, train_disc_fake=0.70 | "
                "val_gen_total=10.2, val_pixel=0.098, val_adv=0.69, val_tv=0.001, val_disc_total=1.38\n"
            )

        # Create a perturbed GT prediction for cGAN chr21
        gt_sp, _ = utils.getMatrixFromNpz(self.gt_dir, "21", pBinSizeInt=self.bin_size)
        gt_arr = gt_sp.toarray().astype(np.float32)
        cgan_arr = np.maximum(0.0, 0.5 * (gt_arr + 0.1 * np.mean(gt_arr)))
        cgan_arr = 0.5 * (cgan_arr + cgan_arr.T)
        sparse.save_npz(os.path.join(cgan_npz_dir, "chr21.npz"), sparse.csr_matrix(cgan_arr))

        # 1) Multi-run leaderboard including Baseline P(d) + cGAN + Diffusion
        lb_out = os.path.join(self.tmp_root, "leaderboard_out")
        df_lb = evaluation.compare_runs_leaderboard(
            search_root_dirs=[cgan_root, diff_root],
            npz_data_dir=self.gt_dir,
            bin_size=self.bin_size,
            chroms=["21"],
            eval_out_dir=lb_out,
            include_baseline_row=True,
        )
        self.assertEqual(len(df_lb), 3)
        self.assertEqual(set(df_lb["Model Tipi"]), {"cGAN", "Diffusion", "Baseline P(d)"})
        self.assertTrue(os.path.exists(os.path.join(lb_out, "multi_run_leaderboard.csv")))
        self.assertTrue(os.path.exists(os.path.join(lb_out, "multi_run_comparison_radar.png")))

        # 2) Head-to-head compare_two_models with fair config check (5-panel without chromatin_dir)
        cmp_out = os.path.join(self.tmp_root, "two_model_cmp")
        res = evaluation.compare_two_models(
            cgan_dir=cgan_root,
            diffusion_dir=diff_root,
            npz_data_dir=self.gt_dir,
            eval_out_dir=cmp_out,
            bin_size=self.bin_size,
            chroms=["21"],
        )
        self.assertTrue(res["fairness_audit"]["fair_comparison"])
        self.assertEqual(res["fairness_audit"]["evaluated_chromosomes"], ["chr21"])
        self.assertEqual(len(res["leaderboard"]), 3)
        for key, fpath in res["files"].items():
            if isinstance(fpath, list):
                for p in fpath:
                    self.assertTrue(os.path.exists(p), f"Missing {key}: {p}")
            else:
                self.assertTrue(os.path.exists(fpath), f"Missing {key}: {fpath}")

        import matplotlib.image as mpimg
        hm5_path = os.path.join(cmp_out, "comparison_heatmaps_chr21.png")
        self.assertIn(hm5_path, res["files"]["heatmap_pngs"])
        w_5panel = mpimg.imread(hm5_path).shape[1]

        # 2b) 6-panel heatmap with chromatin_dir (2D kmer7_revcomp_cosine Input)
        cmp_6_2d = os.path.join(self.tmp_root, "two_model_cmp_6panel_2d")
        res_6_2d = evaluation.compare_two_models(
            cgan_dir=cgan_root,
            diffusion_dir=diff_root,
            npz_data_dir=self.gt_dir,
            eval_out_dir=cmp_6_2d,
            bin_size=self.bin_size,
            chroms=["21"],
            chromatin_dir=self.fp_dir,
            feature_set="representative",
            use_pair_features=True,
        )
        hm6_2d_path = os.path.join(cmp_6_2d, "comparison_heatmaps_chr21.png")
        self.assertTrue(os.path.exists(hm6_2d_path))
        w_6panel_2d = mpimg.imread(hm6_2d_path).shape[1]
        self.assertGreater(w_6panel_2d, w_5panel)

        # 2c) 6-panel heatmap with chromatin_dir and use_pair_features=False (1D bin features fallback)
        cmp_6_1d = os.path.join(self.tmp_root, "two_model_cmp_6panel_1d")
        res_6_1d = evaluation.compare_two_models(
            cgan_dir=cgan_root,
            diffusion_dir=diff_root,
            npz_data_dir=self.gt_dir,
            eval_out_dir=cmp_6_1d,
            bin_size=self.bin_size,
            chroms=["21"],
            chromatin_dir=self.fp_dir,
            feature_set="representative",
            use_pair_features=False,
        )
        hm6_1d_path = os.path.join(cmp_6_1d, "comparison_heatmaps_chr21.png")
        self.assertTrue(os.path.exists(hm6_1d_path))
        w_6panel_1d = mpimg.imread(hm6_1d_path).shape[1]
        self.assertGreater(w_6panel_1d, w_5panel)
        self.assertTrue(os.path.exists(res_6_1d["files"]["leaderboard_csv"]))

        # 2c-bis) Fallback to 1D bin features when use_pair_features=True but get_chrom_pair_matrix fails
        cmp_6_fallback = os.path.join(self.tmp_root, "two_model_cmp_6panel_fallback")
        from unittest import mock
        with mock.patch("hicgan.lib.featurePack.get_chrom_pair_matrix", side_effect=FileNotFoundError("no pair")):
            res_fb = evaluation.compare_two_models(
                cgan_dir=cgan_root,
                diffusion_dir=diff_root,
                npz_data_dir=self.gt_dir,
                eval_out_dir=cmp_6_fallback,
                bin_size=self.bin_size,
                chroms=["21"],
                chromatin_dir=self.fp_dir,
                feature_set="representative",
                use_pair_features=True,
            )
        hm6_fb_path = os.path.join(cmp_6_fallback, "comparison_heatmaps_chr21.png")
        self.assertTrue(os.path.exists(hm6_fb_path))
        self.assertGreater(mpimg.imread(hm6_fb_path).shape[1], w_5panel)

        # 2c-ter) Multi-chromosome comparison (chr1 + chr21)
        gt_1, _ = utils.getMatrixFromNpz(self.gt_dir, "1", pBinSizeInt=self.bin_size)
        sparse.save_npz(os.path.join(cgan_npz_dir, "chr1.npz"), gt_1)
        sparse.save_npz(os.path.join(diff_npz_dir, "chr1.npz"), gt_1)
        cmp_multi = os.path.join(self.tmp_root, "two_model_cmp_multi")
        res_multi = evaluation.compare_two_models(
            cgan_dir=cgan_root,
            diffusion_dir=diff_root,
            npz_data_dir=self.gt_dir,
            eval_out_dir=cmp_multi,
            bin_size=self.bin_size,
            chroms=["1", "21"],
            chromatin_dir=self.fp_dir,
        )
        self.assertEqual(res_multi["fairness_audit"]["evaluated_chromosomes"], ["chr1", "chr21"])
        self.assertEqual(len(res_multi["files"]["heatmap_pngs"]), 2)
        self.assertTrue(os.path.exists(res_multi["files"]["per_chrom_png"]))
        self.assertTrue(os.path.exists(res_multi["files"]["distance_curves_png"]))

        # 2d) Verify plot exception isolation in compare_two_models
        from unittest import mock
        cmp_iso = os.path.join(self.tmp_root, "two_model_cmp_iso")
        with mock.patch("matplotlib.pyplot.subplots", side_effect=RuntimeError("boom")):
            res_iso = evaluation.compare_two_models(
                cgan_dir=cgan_root,
                diffusion_dir=diff_root,
                npz_data_dir=self.gt_dir,
                eval_out_dir=cmp_iso,
                bin_size=self.bin_size,
                chroms=["21"],
            )
        self.assertEqual(res_iso["files"]["heatmap_pngs"], [])
        self.assertIsNone(res_iso["files"]["per_chrom_png"])
        self.assertIsNone(res_iso["files"]["distance_curves_png"])
        self.assertTrue(os.path.exists(res_iso["files"]["leaderboard_csv"]))
        self.assertTrue(os.path.exists(res_iso["files"]["radar_bar_png"]))

        # 3) Edge case: fairness mismatch detection if windowSize differs
        bad_cgan_meta = dict(cgan_meta)
        bad_cgan_meta["windowSize"] = 128
        with open(os.path.join(cgan_train, "hicgan_metadata.pkl"), "wb") as fh:
            pickle.dump(bad_cgan_meta, fh)
        cmp_bad_out = os.path.join(self.tmp_root, "two_model_cmp_mismatch")
        res_bad = evaluation.compare_two_models(
            cgan_dir=cgan_root,
            diffusion_dir=diff_root,
            npz_data_dir=self.gt_dir,
            eval_out_dir=cmp_bad_out,
            bin_size=self.bin_size,
            chroms=["21"],
        )
        self.assertFalse(res_bad["fairness_audit"]["fair_comparison"])
        self.assertEqual(res_bad["fairness_audit"]["field_checks"]["windowSize"]["status"], "mismatch")
        with open(res_bad["files"]["fairness_audit"], "r", encoding="utf-8") as fh:
            audit_json = json.load(fh)
        self.assertFalse(audit_json["fair_comparison"])

    def test_09_mixed_precision_and_resume_epoch_tracking(self):
        # 1) Mixed precision (Keras 3 LossScaleOptimizer) + SpatialTimeEncoding timestep sync on load
        mp_dir = os.path.join(self.tmp_root, "mp_test")
        os.makedirs(mp_dir, exist_ok=True)
        prev_policy = tf.keras.mixed_precision.global_policy()
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            m = hicGAN.HiCGAN(
                log_dir=mp_dir,
                number_factors=92,
                timesteps=20,
                sample_steps=4,
                input_size=64,
                mixed_precision=True,
                use_embedding=True,
                enforce_symmetry=False,
                use_pair_features=True,
            )
            inp = tf.random.normal((2, 192, 92, 1))
            tgt = tf.random.uniform((2, 64, 64, 1), 0.0, 1.0)
            tr_losses = m.train_step(inp, tgt)
            va_losses = m.validationStep(inp, tgt)
            pr_out = m.predictionStep(inp, training=False)
            for v in tr_losses + va_losses:
                self.assertTrue(np.isfinite(float(v)))
            self.assertEqual(tuple(pr_out.shape), (2, 64, 64, 1))
            save_p = os.path.join(mp_dir, "denoiser_best.keras")
            m.denoiser.save(save_p)
        finally:
            tf.keras.mixed_precision.set_global_policy(prev_policy)

        # Reload with mismatched default timesteps=1000 and verify auto-sync -> 20
        m_reload = hicGAN.HiCGAN(
            log_dir=mp_dir,
            number_factors=92,
            timesteps=1000,
            sample_steps=4,
            input_size=64,
            mixed_precision=False,
            use_embedding=True,
            enforce_symmetry=False,
            use_pair_features=True,
        )
        m_reload.load_denoiser(save_p)
        self.assertEqual(m_reload.timesteps, 20)
        pr_re = m_reload.predictionStep(inp, training=False)
        self.assertEqual(tuple(pr_re.shape), (2, 64, 64, 1))

        # 2) --resume exact epoch tracking even when plotFrequency > epochs
        res_dir = os.path.join(self.tmp_root, "resume_test")
        hic_training.main([
            "--dataDir", self.gt_dir,
            "--trainingChromatinFolders", self.fp_dir,
            "--validationChromatinFolders", self.fp_dir,
            "--splitFile", self.split_path,
            "--outputFolder", res_dir,
            "--binSize", str(self.bin_size),
            "--windowSize", "64",
            "--windowMode", "grid",
            "--gridStride", "16",
            "--inputMode", "feature_pack",
            "--featureSet", "representative",
            "--targetNorm", "p99_log10",
            "--flankPadChromosomes", "all",
            "--epochs", "2",
            "--batchSize", "4",
            "--timesteps", "12",
            "--sampleSteps", "3",
            "--plotFrequency", "10",
            "--keepTFRecords",
            "--seed", "42",
        ])
        # Resume for 1 more epoch (total 3)
        hic_training.main([
            "--outputFolder", res_dir,
            "--epochs", "3",
            "--batchSize", "4",
            "--timesteps", "12",
            "--sampleSteps", "3",
            "--plotFrequency", "10",
            "--resume",
            "--trainOnly",
        ])
        df_loss = evaluation.parse_training_losses(res_dir)
        self.assertEqual(list(df_loss["epoch"]), [1, 2, 3])

    def test_10_random2d_diagonal_colab_layout_and_eval_cli(self):
        import subprocess

        # 1) random2d + diagonal + predict_band
        r2d_train = os.path.join(self.tmp_root, "r2d_train")
        r2d_pred = os.path.join(self.tmp_root, "r2d_pred")
        hic_training.main([
            "--dataDir", self.gt_dir,
            "--splitFile", self.split_path,
            "--outputFolder", r2d_train,
            "--inputMode", "random2d",
            "--windowMode", "diagonal",
            "--windowSize", "64",
            "--epochs", "1",
            "--batchSize", "4",
            "--timesteps", "10",
            "--sampleSteps", "2",
            "--plotFrequency", "0",
            "--seed", "42",
        ])
        hic_predict.main([
            "--trainedModel", r2d_train,
            "--dataDir", self.gt_dir,
            "--splitFile", self.split_path,
            "--split", "test",
            "--outputFolder", r2d_pred,
            "--inputMode", "random2d",
            "--windowMode", "diagonal",
            "--windowSize", "64",
            "--batchSize", "4",
            "--chromSizesPath", self.gt_dir,
        ])
        r2d_npz = os.path.join(r2d_pred, "npz_matrices", "chr21.npz")
        self.assertTrue(os.path.exists(r2d_npz))
        r2d_arr = sparse.load_npz(r2d_npz).toarray()
        np.testing.assert_allclose(r2d_arr, r2d_arr.T, atol=1e-6)

        # 2) Phase 1 Colab directory layout (OUTPUT_DIR/test_eval/npz_matrices) + CLI subprocess
        if not hasattr(self.__class__, "diff_pred_dir"):
            self.test_07_e2e_training_predict_eval_and_phased_modes()

        colab_cgan = os.path.join(self.tmp_root, "colab_cgan_layout")
        colab_cgan_eval = os.path.join(colab_cgan, "test_eval")
        colab_cgan_npz = os.path.join(colab_cgan_eval, "npz_matrices")
        os.makedirs(colab_cgan_npz, exist_ok=True)
        shutil.copy2(os.path.join(self.diff_train_dir, "target_value_range.json"), os.path.join(colab_cgan, "target_value_range.json"))
        with open(os.path.join(self.diff_train_dir, "hicgan_metadata.pkl"), "rb") as fh:
            cmeta = dict(pickle.load(fh))
        cmeta["modelType"] = "cgan"
        cmeta.pop("timesteps", None)
        cmeta.pop("sampleSteps", None)
        with open(os.path.join(colab_cgan, "hicgan_metadata.pkl"), "wb") as fh:
            pickle.dump(cmeta, fh)
        with open(os.path.join(colab_cgan_eval, "predictions.pkl"), "wb") as fh:
            pickle.dump({"blendMode": "raised_cosine_peak", "binSize": self.bin_size, "windowSize": 64}, fh)
        with open(os.path.join(colab_cgan, "training.log"), "w", encoding="utf-8") as fh:
            fh.write(
                "[EPOCH-LOSS] epoch=1/2 | train_gen_total=12.5, train_pixel=0.120, train_adv=0.69, train_tv=0.001, "
                "train_disc_total=1.38, train_disc_true=0.69, train_disc_fake=0.69 | "
                "val_gen_total=11.8, val_pixel=0.115, val_adv=0.68, val_tv=0.001, val_disc_total=1.37\n"
            )
        gt_sp, _ = utils.getMatrixFromNpz(self.gt_dir, "21", pBinSizeInt=self.bin_size)
        sparse.save_npz(os.path.join(colab_cgan_npz, "chr21.npz"), gt_sp)

        diff_root = os.path.join(self.tmp_root, "diff_cli_run")
        os.makedirs(os.path.join(diff_root, "train_run"), exist_ok=True)
        os.makedirs(os.path.join(diff_root, "predict_run", "npz_matrices"), exist_ok=True)
        shutil.copy2(os.path.join(self.diff_train_dir, "hicgan_metadata.pkl"), os.path.join(diff_root, "train_run", "hicgan_metadata.pkl"))
        shutil.copy2(os.path.join(self.diff_train_dir, "target_value_range.json"), os.path.join(diff_root, "train_run", "target_value_range.json"))
        shutil.copy2(os.path.join(self.diff_train_dir, "training.log"), os.path.join(diff_root, "train_run", "training.log"))
        shutil.copy2(os.path.join(self.diff_pred_dir, "predictions.pkl"), os.path.join(diff_root, "predict_run", "predictions.pkl"))
        shutil.copy2(os.path.join(self.diff_pred_dir, "npz_matrices", "chr21.npz"), os.path.join(diff_root, "predict_run", "npz_matrices", "chr21.npz"))

        cli_out = os.path.join(self.tmp_root, "cli_cmp_out")
        proc = subprocess.run(
            [
                sys.executable, "-m", "hicgan.lib.evaluation",
                "--cganDir", colab_cgan,
                "--diffusionDir", diff_root,
                "--gtDir", self.gt_dir,
                "--outDir", cli_out,
                "--chroms", "21",
                "--binSize", str(self.bin_size),
                "--chromatinDir", self.fp_dir,
                "--featureSet", "representative",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertTrue(os.path.exists(os.path.join(cli_out, "comparison_heatmaps_chr21.png")))
        with open(os.path.join(cli_out, "fairness_audit.json"), "r", encoding="utf-8") as fh:
            fa = json.load(fh)
        self.assertTrue(fa["fair_comparison"], msg=str(fa))
        self.assertEqual(fa["field_checks"]["target_range_hi"]["status"], "match")
        self.assertEqual(fa["field_checks"]["filterEmptyWindows"]["status"], "match")

        # Also verify resolving directly from test_eval subdir retains parent train_dir metadata
        res_sub = evaluation.compare_two_models(
            cgan_dir=colab_cgan_eval,
            diffusion_dir=diff_root,
            npz_data_dir=self.gt_dir,
            eval_out_dir=os.path.join(self.tmp_root, "cli_cmp_sub"),
            bin_size=self.bin_size,
            chroms=["21"],
            chromatin_dir=self.fp_dir,
            feature_set="representative",
            use_pair_features=True,
        )
        self.assertTrue(res_sub["fairness_audit"]["fair_comparison"])
        self.assertTrue(os.path.exists(os.path.join(self.tmp_root, "cli_cmp_sub", "comparison_heatmaps_chr21.png")))

        # 3) Validate colab_rna_diffusion_clean.ipynb repo URL and paths
        nb_path = os.path.join(REPO_ROOT, "colab_rna_diffusion_clean.ipynb")
        with open(nb_path, "r", encoding="utf-8") as fh:
            nb = json.load(fh)
        self.assertEqual(nb["nbformat"], 4)
        nb_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
        self.assertIn("https://github.com/nehirvardar/Hi-cGAN.git", nb_src)
        self.assertIn("data/feature_pack", nb_src)
        self.assertNotIn("hicgan/data/feature_pack", nb_src)
        self.assertIn("test_eval", nb_src)
        self.assertIn("chromatin_dir=CHROMATIN_DIR", nb_src)
        self.assertIn("feature_set=FEATURE_SET", nb_src)
        self.assertIn("use_pair_features=USE_PAIR_FEATURES", nb_src)


if __name__ == "__main__":
    unittest.main()
