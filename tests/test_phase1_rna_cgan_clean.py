import json
import os
import shutil
import tempfile
import unittest
import matplotlib
matplotlib.use("Agg")
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from hicgan.lib import utils, featurePack, dataContainer, records, hicGAN, evaluation
from hicgan import training as hic_training
from hicgan import predict as hic_predict


def test_splits_file_integrity_and_no_leakage():
    splits = utils.parseSplitFile("hicgan/data/splits.txt")
    assert len(splits["train"]) == 12
    assert len(splits["val"]) == 3
    assert len(splits["test"]) == 8

    expected_train = {"2", "4", "6", "12", "13", "15", "16", "18", "20", "21", "22", "X"}
    expected_val = {"8", "10", "14"}
    expected_test = {"1", "3", "5", "7", "9", "11", "17", "19"}

    assert set(splits["train"]) == expected_train
    assert set(splits["val"]) == expected_val
    assert set(splits["test"]) == expected_test

    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])


def test_p99_log10_scaling_reversibility_and_train_only_isolation():
    rng = np.random.default_rng(123)
    raw = np.zeros((32, 32), dtype=np.float64)
    raw[0:10, 0:10] = rng.uniform(0.1, 15.0, size=(10, 10))
    raw = 0.5 * (raw + raw.T)

    hi = float(np.percentile(np.log1p(10.0 * raw[raw > 0]), 99.0))
    norm_dense = utils.toP99LogUnitRange(raw, hi, scale_factor=10.0)
    assert norm_dense.min() >= 0.0
    assert norm_dense.max() <= 1.0
    assert np.all(norm_dense[raw == 0] == 0.0)

    # Unclipped elements must invert with near-zero error
    unclipped_mask = np.log1p(10.0 * raw) <= hi
    recovered_dense = utils.fromP99LogUnitRange(norm_dense, hi, scale_factor=10.0)
    max_err = np.max(np.abs(recovered_dense[unclipped_mask] - raw[unclipped_mask]))
    assert max_err < 1e-6

    # Sparse CSR round-trip
    raw_csr = sp.csr_matrix(raw)
    norm_csr = utils.toP99LogUnitRange(raw_csr, hi, scale_factor=10.0)
    rec_csr = utils.fromP99LogUnitRange(norm_csr, hi, scale_factor=10.0)
    assert np.max(np.abs(rec_csr.toarray()[unclipped_mask] - raw[unclipped_mask])) < 1e-6

    # Edge case: all-zero array
    zero_arr = np.zeros((16, 16), dtype=np.float32)
    assert np.all(utils.toP99LogUnitRange(zero_arr, hi) == 0.0)
    assert np.all(utils.fromP99LogUnitRange(zero_arr, hi) == 0.0)

    # Verify train-only isolation (val/test chromosomes never affect hi)
    with tempfile.TemporaryDirectory() as tmpdir:
        c21 = sp.csr_matrix( np.diag([1.0, 2.0, 5.0, 10.0]) )
        c8_val_extreme = sp.csr_matrix( np.diag([1e6, 2e6, 5e6, 1e7]) )
        sp.save_npz(os.path.join(tmpdir, "chr21.npz"), c21)
        sp.save_npz(os.path.join(tmpdir, "chr8.npz"), c8_val_extreme)

        range_json = os.path.join(tmpdir, "target_value_range.json")
        cfg = utils.computeGlobalP99TargetRange([tmpdir], ["21"], scale_factor=10.0, clip_percentile=99.0, pSavePath=range_json)
        expected_hi = float(np.percentile(np.log1p(10.0 * np.array([1.0, 2.0, 5.0, 10.0])), 99.0))
        assert np.isclose(cfg["hi"], expected_hi, atol=1e-6)

        loaded_cfg = utils.loadTargetRangeConfig(range_json)
        assert loaded_cfg["method"] == "p99_log10"
        assert np.isclose(loaded_cfg["hi"], expected_hi, atol=1e-6)


def test_feature_pack_and_datacontainer_chr21_sat_filter():
    # Both alias paths should work
    assert featurePack.is_feature_pack_source("data/feature_pack")
    assert featurePack.is_feature_pack_source("data/feature_paketi_20261029")

    # Load chr21 (187 bins < 192 required for W=64, flanking=64)
    dc = dataContainer.DataContainer(
        chromosome="21",
        matrixFilePath="hicgan/data/rna_gt",
        chromatinFolder="data/feature_paketi_20261029",
        binSize=250000,
        inputMode="feature_pack",
        flankPadChromosomes="all",
        featureSet="representative",
        usePairFeatures=True,
        filterEmptyWindows=True,
    )
    p99_cfg = {"method": "p99_log10", "hi": 5.0, "scale_factor": 10.0}
    dc.loadData(
        windowSize=64,
        flankingSize=64,
        windowMode="grid",
        gridStride=16,
        targetValueRange=p99_cfg,
        flankPadChromosomes="all",
        featureSet="representative",
        usePairFeatures=True,
        filterEmptyWindows=True,
    )

    assert dc.original_num_bins == 187
    assert dc.left_padded_bins == 64
    assert dc.right_padded_bins == 64
    assert dc.numBins == 187 + 128
    assert dc.hasEnoughBins()

    total_grid = dc.gridCoordinates.shape[0]
    kept_indices = set(dc.sampleIndices.tolist())
    assert 0 < len(kept_indices) <= total_grid

    # Also test on chr22 (204 bins) which has empty off-diagonal windows that MUST be filtered out by SAT
    dc22 = dataContainer.DataContainer(
        chromosome="22",
        matrixFilePath="hicgan/data/rna_gt",
        chromatinFolder="data/feature_paketi_20261029",
        binSize=250000,
        inputMode="feature_pack",
        flankPadChromosomes="all",
        featureSet="representative",
        usePairFeatures=True,
        filterEmptyWindows=True,
    )
    dc22.loadData(
        windowSize=64,
        flankingSize=64,
        windowMode="grid",
        gridStride=16,
        targetValueRange=p99_cfg,
        flankPadChromosomes="all",
        featureSet="representative",
        usePairFeatures=True,
        filterEmptyWindows=True,
    )
    total_grid_22 = dc22.gridCoordinates.shape[0]
    kept_22 = set(dc22.sampleIndices.tolist())
    assert 0 < len(kept_22) < total_grid_22
    dense_gt_22 = dc22.sparseHiCMatrix.toarray()
    for raw_idx in range(total_grid_22):
        r0, c0 = int(dc22.gridCoordinates[raw_idx, 0]), int(dc22.gridCoordinates[raw_idx, 1])
        tile = dense_gt_22[r0 + 64 : r0 + 128, c0 + 64 : c0 + 128]
        tile_nnz = int(np.sum(tile > 0))
        is_diag = (r0 == c0)
        if is_diag or tile_nnz > 0:
            assert raw_idx in kept_22
        else:
            assert raw_idx not in kept_22

    # Verify SAT filter exactness on every single grid coordinate of chr21
    dense_gt = dc.sparseHiCMatrix.toarray()
    W = dc.windowSize
    flank = dc.flankingSize
    for raw_idx in range(total_grid):
        r0, c0 = int(dc.gridCoordinates[raw_idx, 0]), int(dc.gridCoordinates[raw_idx, 1])
        tile = dense_gt[r0 + flank : r0 + flank + W, c0 + flank : c0 + flank + W]
        tile_nnz = int(np.sum(tile > 0))
        is_diag = (r0 == c0)
        if is_diag or tile_nnz > 0:
            assert raw_idx in kept_indices, f"Window {raw_idx} (diag={is_diag}, nnz={tile_nnz}) should be kept!"
        else:
            assert raw_idx not in kept_indices, f"Empty off-diagonal window {raw_idx} should be filtered out!"

    sample0 = dc.getSampleData(0)
    # 27 1D features -> 2*(27 + 1) + 64 = 120 channels, 3*64 = 192 length
    assert sample0["factorData"].shape == (192, 120, 1)
    assert sample0["out_matrixData"].shape == (64, 64, 1)
    assert sample0["out_matrixData"].min() >= 0.0
    assert sample0["out_matrixData"].max() <= 1.0

    # Verify mirror_function symmetry on sample0
    f_t = tf.constant(sample0["factorData"], dtype=tf.float32)
    m_t = tf.constant(sample0["out_matrixData"], dtype=tf.float32)
    f_mir, m_mir = records.mirror_function(f_t, m_t, is_2d_input=False, is_grid_mode=True, has_pair_2d=True)
    # Applying mirror_function twice must recover the exact original tensors
    f_mir2, m_mir2 = records.mirror_function(
        f_mir["factorData"], m_mir["out_matrixData"], is_2d_input=False, is_grid_mode=True, has_pair_2d=True
    )
    assert np.allclose(sample0["factorData"], f_mir2["factorData"].numpy(), atol=1e-6)
    assert np.allclose(sample0["out_matrixData"], m_mir2["out_matrixData"].numpy(), atol=1e-6)


def test_hicgan_condition_encoder_dropout_and_deterministic_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        scope = tf.distribute.OneDeviceStrategy(device="/CPU:0")
        with scope.scope():
            # 1) Default dropout_rate=0.0 -> no Dropout layers in Generator
            model_nodrop = hicGAN.HiCGAN(
                log_dir=tmpdir,
                number_factors=120,
                input_size=64,
                dropout_rate=0.0,
                underpred_boost_factor=1.5,
                decay_steps=100,
                scope=scope,
                use_embedding=True,
                enforce_symmetry=False,
                use_pair_features=True,
            )
            assert model_nodrop.condition_encoder is not None
            dummy_x = tf.random.normal((2, 192, 120))
            dummy_y = tf.random.uniform((2, 64, 64, 1), minval=0.0, maxval=1.0)
            emb_out = model_nodrop.condition_encoder(dummy_x)
            assert emb_out.shape == (2, 64, 64, 16)

            # Check CosineDecay schedule (Keras 2 uses .learning_rate, Keras 3 stores schedule in ._learning_rate)
            lr_sched = getattr(model_nodrop.generator_optimizer, "_learning_rate", model_nodrop.generator_optimizer.learning_rate)
            assert isinstance(
                lr_sched,
                tf.keras.optimizers.schedules.CosineDecay,
            )

            # Check underpred_boost_factor in generator_loss
            target = tf.ones((1, 4, 4, 1), dtype=tf.float32) * 0.5
            gen_under = tf.ones((1, 4, 4, 1), dtype=tf.float32) * 0.3  # diff = 0.2 (under)
            gen_over = tf.ones((1, 4, 4, 1), dtype=tf.float32) * 0.7   # diff = 0.2 (over)
            disc_out = tf.zeros((1, 4, 4, 1), dtype=tf.float32)
            _, _, pix_under, _ = model_nodrop.generator_loss(disc_out, gen_under, target)
            _, _, pix_over, _ = model_nodrop.generator_loss(disc_out, gen_over, target)
            assert np.isclose(float(pix_under) / float(pix_over), 1.5, rtol=1e-5)

            # 2) With dropout_rate=0.5, validationStep MUST still be deterministic (training=False)
            model_drop = hicGAN.HiCGAN(
                log_dir=tmpdir,
                number_factors=120,
                input_size=64,
                dropout_rate=0.5,
                underpred_boost_factor=1.5,
                decay_steps=100,
                scope=scope,
                use_embedding=True,
                enforce_symmetry=False,
                use_pair_features=True,
            )
            val1 = model_drop.validationStep(dummy_x, dummy_y)
            val2 = model_drop.validationStep(dummy_x, dummy_y)
            for t1, t2 in zip(val1, val2):
                assert np.allclose(t1.numpy(), t2.numpy(), atol=1e-7)


def test_end_to_end_smoke_train_predict_and_eval():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        train_out = os.path.join(tmpdir, "smoke_train")
        pred_out = os.path.join(tmpdir, "smoke_pred")

        # 2-epoch smoke training on chr21 (train) and chr22 (val)
        hic_training.main([
            "-tm", "hicgan/data/rna_gt",
            "-vm", "hicgan/data/rna_gt",
            "-tchroms", "21",
            "-vchroms", "22",
            "-tcp", "data/feature_paketi_20261029",
            "-vcp", "data/feature_paketi_20261029",
            "-im", "feature_pack",
            "-fset", "representative",
            "-upf",
            "-wm", "grid",
            "-gs", "64",
            "-ws", "64",
            "-ep", "2",
            "-bs", "16",
            "-o", train_out,
            "-seed", "42",
        ])

        assert os.path.exists(os.path.join(train_out, "target_value_range.json"))
        assert os.path.exists(os.path.join(train_out, "hicgan_metadata.pkl"))
        assert os.path.exists(os.path.join(train_out, "generator_best.keras"))
        assert os.path.exists(os.path.join(train_out, "generator_final.keras"))

        with open(os.path.join(train_out, "target_value_range.json"), "r") as f:
            tvr = json.load(f)
        assert tvr["method"] == "p99_log10"
        assert tvr["scale_factor"] == 10.0
        assert tvr["hi"] > 0.0

        # Predict on chr22 and run unified evaluation
        hic_predict.main([
            "-trm", os.path.join(train_out, "generator_best.keras"),
            "-tcp", "data/feature_paketi_20261029",
            "-csp", "hicgan/data/rna_gt",
            "-im", "feature_pack",
            "-fset", "representative",
            "-upf",
            "-wm", "grid",
            "-gs", "64",
            "-ws", "64",
            "-pc", "22",
            "-o", pred_out,
            "-eval",
            "-gt", "hicgan/data/rna_gt",
        ])

        pred_npz = os.path.join(pred_out, "npz_matrices", "chr22.npz")
        scorecard_csv = os.path.join(pred_out, "evaluation", "core_metrics_scorecard.csv")
        triple_png = os.path.join(pred_out, "evaluation", "heatmap_triple_chr22.png")
        sample_png = os.path.join(train_out, "testpred_epoch_00000.png")
        assert os.path.exists(pred_npz)
        assert os.path.exists(scorecard_csv)
        assert os.path.exists(triple_png)
        assert os.path.exists(sample_png)

        mat = sp.load_npz(pred_npz).toarray()
        assert mat.shape == (204, 204)
        assert not np.any(np.isnan(mat))
        assert np.allclose(mat, mat.T, atol=1e-5)

        # Verify compare_runs_leaderboard works directly on hicPredict outputFolder
        df_lb = evaluation.compare_runs_leaderboard(
            search_root_dirs=[pred_out],
            npz_data_dir="hicgan/data/rna_gt",
            chroms=["22"],
            eval_out_dir=os.path.join(pred_out, "leaderboard"),
        )
        assert not df_lb.empty
        assert os.path.exists(os.path.join(pred_out, "leaderboard", "multi_run_leaderboard.csv"))


def test_phased_cdo_trainonly_and_phased_predict_and_windows_paths():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        train_out = os.path.join(tmpdir, "phased_train")
        pred_out = os.path.join(tmpdir, "phased_pred")

        # Use an absolute Windows path with backslashes/drive-letter for -tcp to verify
        # writeTFRecord sanitizes folderName and writes into pOutputFolder
        abs_fp_win = os.path.abspath("data/feature_pack").replace("/", "\\")

        # Phase 1A: --createDataOnly (-cdo)
        hic_training.main([
            "-tm", "hicgan/data/rna_gt",
            "-vm", "hicgan/data/rna_gt",
            "-tchroms", "21",
            "-vchroms", "22",
            "-tcp", abs_fp_win,
            "-vcp", abs_fp_win,
            "-im", "feature_pack",
            "-fset", "direct",
            "-npf",
            "-wm", "grid",
            "-gs", "64",
            "-ws", "64",
            "-cdo",
            "-o", train_out,
            "-seed", "42",
        ])
        assert os.path.exists(os.path.join(train_out, "hicgan_metadata.pkl"))
        assert os.path.exists(os.path.join(train_out, "target_value_range.json"))
        # Verify TFRecords were written inside train_out (not discarded by Windows drive path)
        tfrecs = [f for f in os.listdir(train_out) if f.endswith(".tfrecord")]
        assert len(tfrecs) >= 2

        # Phase 1B: --trainOnly (-to), passing --targetValueRange to verify double-scaling is prevented
        # and omitting -tm/-vm/-tcp/-vcp/-tchroms/-vchroms so they are loaded from metadata
        hic_training.main([
            "-to",
            "-tvr", os.path.join(train_out, "target_value_range.json"),
            "-ep", "1",
            "-bs", "16",
            "-k",
            "-o", train_out,
            "-seed", "42",
        ])
        assert os.path.exists(os.path.join(train_out, "generator_best.keras"))

        # Phase 2A: predict.py --mode create-data with blendMode='uniform'
        hic_predict.main([
            "--mode", "create-data",
            "-trm", train_out,
            "-tcp", abs_fp_win,
            "-csp", "hicgan/data/rna_gt",
            "-wm", "grid",
            "-gs", "64",
            "-ws", "64",
            "-bm", "uniform",
            "-pc", "21",
            "-o", pred_out,
        ])
        assert os.path.exists(os.path.join(pred_out, "prediction_vars.pkl"))

        # Phase 2B: predict.py --mode predict (without repeating -pc or -bm)
        hic_predict.main([
            "--mode", "predict",
            "-trm", train_out,
            "-o", pred_out,
        ])
        assert os.path.exists(os.path.join(pred_out, "predictions.pkl"))

        # Phase 2C: predict.py --mode make-matrix (CPU-only, without -trm, -pc, -tvr, or -bm)
        hic_predict.main([
            "--mode", "make-matrix",
            "-o", pred_out,
        ])
        pred_chr21 = os.path.join(pred_out, "npz_matrices", "chr21.npz")
        assert os.path.exists(pred_chr21)
        mat21 = sp.load_npz(pred_chr21).toarray()
        assert mat21.shape == (187, 187)
        assert np.allclose(mat21, mat21.T, atol=1e-5)

        with open(os.path.join(pred_out, "predictions.pkl"), "rb") as pf:
            import pickle
            saved_p = pickle.load(pf)
        assert saved_p["blendMode"] == "uniform"
        assert isinstance(saved_p["targetValueRange"], dict)
        assert saved_p["targetValueRange"]["method"] == "p99_log10"


def test_blend_modes_and_splitfile_cli():
    rng = np.random.default_rng(7)
    W = 16
    M = 32
    coords = np.array([[0, 0], [0, 8], [0, 16], [8, 8], [8, 16], [16, 16]], dtype=np.int64)
    preds = rng.uniform(0.0, 0.9, size=(len(coords), W, W)).astype(np.float32)

    m_peak = utils.rebuildMatrixGrid(preds, coords, W, M, blend_mode="raised_cosine_peak").toarray()
    m_unif = utils.rebuildMatrixGrid(preds, coords, W, M, blend_mode="uniform").toarray()

    assert m_peak.shape == (M, M)
    assert m_unif.shape == (M, M)
    assert np.allclose(m_peak, m_peak.T, atol=1e-6)
    assert np.allclose(m_unif, m_unif.T, atol=1e-6)
    assert not np.allclose(m_peak, m_unif)  # Peak-weighted raised-cosine differs from uniform mean

    with unittest.TestCase().assertRaises(ValueError):
        utils.rebuildMatrixGrid(preds, coords, W, M, blend_mode="invalid_mode")


def test_loss_modes_and_no_logp_norm_toggle():
    import pickle

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        # 1) All 3 --lossMode presets resolve exact weights
        for mode_name, exp_fg, exp_upb, exp_bg in (
            ("current", 10.0, 1.5, 0.0),
            ("paper", 0.0, 1.0, 0.0),
            ("balanced", 2.5, 1.0, 0.5),
        ):
            parsed = hic_training.parse_arguments().parse_args([
                "-o", os.path.join(tmpdir, f"lm_{mode_name}"),
                "--lossMode", mode_name,
            ])
            cfg = hic_training.resolve_loss_mode_params(parsed)
            assert cfg["lossMode"] == mode_name
            assert abs(cfg["foregroundLossWeight"] - exp_fg) < 1e-6
            assert abs(cfg["underpredBoost"] - exp_upb) < 1e-6
            assert abs(cfg["bgL1Weight"] - exp_bg) < 1e-6

        # 2) Verify paper vs balanced pixel loss mechanics
        m_paper = hicGAN.HiCGAN(
            log_dir=os.path.join(tmpdir, "m_paper"),
            number_factors=120,
            input_size=64,
            loss_mode="paper",
            foreground_loss_weight=0.0,
            underpred_boost_factor=1.0,
            bg_l1_weight=0.0,
            use_embedding=True,
            use_pair_features=True,
        )
        m_bal = hicGAN.HiCGAN(
            log_dir=os.path.join(tmpdir, "m_bal"),
            number_factors=120,
            input_size=64,
            loss_mode="balanced",
            foreground_loss_weight=2.5,
            underpred_boost_factor=1.0,
            bg_l1_weight=0.5,
            use_embedding=True,
            use_pair_features=True,
        )
        disc_zero = tf.zeros((2, 6, 6, 1), dtype=tf.float32)
        tar_zero = tf.zeros((2, 64, 64, 1), dtype=tf.float32)
        gen_half = tf.fill((2, 64, 64, 1), 0.5)
        _, _, pix_paper, _ = m_paper.generator_loss(disc_zero, gen_half, tar_zero)
        _, _, pix_bal, _ = m_bal.generator_loss(disc_zero, gen_half, tar_zero)
        assert abs(float(pix_paper) - 0.5) < 1e-5
        assert abs(float(pix_bal) - 0.75) < 1e-5  # 0.5 + 0.5 * 0.5

        # 3) Verify --noLogpNorm + --lossMode balanced E2E (raw counts, ReLU Generator)
        raw_train = os.path.join(tmpdir, "raw_cgan_train")
        raw_pred = os.path.join(tmpdir, "raw_cgan_pred")
        hic_training.main([
            "-tm", "hicgan/data/rna_gt",
            "-vm", "hicgan/data/rna_gt",
            "-tchroms", "21",
            "-vchroms", "22",
            "-tcp", "data/feature_pack",
            "-vcp", "data/feature_pack",
            "-im", "feature_pack",
            "-fset", "representative",
            "-upf",
            "-wm", "grid",
            "-gs", "64",
            "-ws", "64",
            "--lossMode", "balanced",
            "--noLogpNorm",
            "-ep", "1",
            "-bs", "4",
            "--plotFrequency", "0",
            "-o", raw_train,
            "-seed", "42",
        ])
        with open(os.path.join(raw_train, "target_value_range.json"), "r", encoding="utf-8") as fh:
            tvr = json.load(fh)
        assert tvr["method"] == "none"
        with open(os.path.join(raw_train, "hicgan_metadata.pkl"), "rb") as fh:
            meta = pickle.load(fh)
        assert meta["targetNorm"] == "none"
        assert meta["logpNorm"] is False
        assert meta["rawTargetMode"] is True
        assert meta["lossMode"] == "balanced"
        assert abs(meta["bgL1Weight"] - 0.5) < 1e-6

        hic_predict.main([
            "-trm", os.path.join(raw_train, "generator_best.keras"),
            "-tcp", "data/feature_pack",
            "-csp", "hicgan/data/rna_gt",
            "-im", "feature_pack",
            "-fset", "representative",
            "-upf",
            "-wm", "grid",
            "-gs", "64",
            "-ws", "64",
            "-pc", "22",
            "-bs", "4",
            "-o", raw_pred,
        ])
        assert os.path.exists(os.path.join(raw_pred, "npz_matrices", "chr22.npz"))

        # 4) Verify colab_rna_cgan_clean.ipynb has LOSS_MODE and USE_LOGP_NORM flags
        with open("colab_rna_cgan_clean.ipynb", "r", encoding="utf-8") as fh:
            nb = json.load(fh)
        assert nb["nbformat"] == 4
        nb_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
        assert "LOSS_MODE" in nb_src
        assert "USE_LOGP_NORM" in nb_src
        assert "--noLogpNorm" in nb_src


class TestPhase1RnaCganClean(unittest.TestCase):
    def test_01_splits_file_integrity_and_no_leakage(self):
        test_splits_file_integrity_and_no_leakage()

    def test_02_p99_log10_scaling_reversibility_and_train_only_isolation(self):
        test_p99_log10_scaling_reversibility_and_train_only_isolation()

    def test_03_feature_pack_and_datacontainer_chr21_sat_filter(self):
        test_feature_pack_and_datacontainer_chr21_sat_filter()

    def test_04_hicgan_condition_encoder_dropout_and_deterministic_validation(self):
        test_hicgan_condition_encoder_dropout_and_deterministic_validation()

    def test_05_end_to_end_smoke_train_predict_and_eval(self):
        test_end_to_end_smoke_train_predict_and_eval()

    def test_06_blend_modes_and_splitfile_cli(self):
        test_blend_modes_and_splitfile_cli()

    def test_07_phased_cdo_trainonly_and_phased_predict_and_windows_paths(self):
        test_phased_cdo_trainonly_and_phased_predict_and_windows_paths()

    def test_08_loss_modes_and_no_logp_norm_toggle(self):
        test_loss_modes_and_no_logp_norm_toggle()


if __name__ == "__main__":
    unittest.main(verbosity=2)
