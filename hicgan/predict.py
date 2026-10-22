import argparse
import multiprocessing
import numpy as np
import os
import csv
import tensorflow as tf
from scipy import sparse
from .lib import dataContainer
from .lib import records
from .lib import hicGAN
from .lib import utils
import logging
from hicgan._version import __version__
import pickle

log = logging.getLogger(__name__)

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainedModel", "-trm", required=False,
                        type=str,
                        help="Trained generator model to predict from")
    parser.add_argument("--predictionChromosomesFolders", "-tcp", required=False,
                        type=str, default=None,
                        help="Path where test data (bigwig files) resides, or path to .npz folder / chrom_index.csv in random2d mode")
    parser.add_argument("--chromSizesPath", "-csp", required=False,
                        type=str, default=None,
                        help="Path to chrom_index.csv, .npz directory, or .npz file to determine chromosome bin counts in 'random2d' mode")
    parser.add_argument("--inputMode", "-im", required=False,
                        type=str, choices=["bigwig", "random2d"], default=None,
                        help="Input feature mode: 'bigwig' (1D chromatin BigWig files) or 'random2d' (direct 2D symmetric random matrix). "
                             "Defaults to 'random2d' when -tcp is omitted or points to .npz/chrom_index.csv.")
    parser.add_argument("--windowMode", "-wm", required=False,
                        type=str, choices=["diagonal", "grid"], default="diagonal",
                        help="Windowing mode: 'diagonal' (sliding windows along main diagonal) or 'grid' (2D grid over upper triangle [i:i+W, j:j+W]).")
    parser.add_argument("--gridStride", "-gs", required=False,
                        type=int, default=None,
                        help="Stride in bins for 'grid' windowMode (defaults to windowSize // 2).")
    parser.add_argument("--seed", "-sd", required=False,
                        type=int, default=42,
                        help="Random seed for deterministic 2D random input matrix generation in 'random2d' mode (default: 42).")
    parser.add_argument("--logTransformTarget", "-ltt", required=False,
                        action="store_true", default=False,
                        help="Apply expm1(x) after rescaling from [0, 1] (automatically enabled if target_value_range.json has log_transform=true).")
    parser.add_argument("--npzOutputFolder", "-no", required=False,
                        type=str, default=None,
                        help="Directory where predicted per-chromosome chr{X}.npz files and chrom_index.csv will be written (default: <outputFolder>/npz_matrices).")
    parser.add_argument("--predictionChromosomes", "-pc", required=False,
                        type=str,
                        nargs='+',
                        help="Chromosomes the matrix should be predicted for.")
    parser.add_argument("--matrixOutputName", "-mn", required=False,
                        type=str,
                        default="predMatrix.cool",
                        help="Name of the output cool-file")
    parser.add_argument("--parameterOutputFile", "-pf", required=False,
                        type=str,
                        default="predParams.csv",
                        help="Name of the parameter file")
    parser.add_argument("--outputFolder", "-o", required=False,
                        type=str,
                        default="./", 
                        help="Output path for predicted cool-file and npz matrices")
    parser.add_argument("--multiplier", "-mul", required=False,
                        type=int, 
                        default=10, 
                        help="Multiplier for scaling the predicted coolers")
    parser.add_argument("--binSize", "-b", required=False,
                        type=int, default=250000,
                        help="Bin size in bp (default: 250000)")
    parser.add_argument("--batchSize", "-bs", required=False,
                        type=int,
                        default=32, 
                        help="Batch size for predicting")
    parser.add_argument("--windowSize", "-ws", required=False,
                        type=int,
                        choices=[64, 128, 256, 512, 768, 1024],
                        help="Window size for predicting; must be the same as in trained model.")
    parser.add_argument("--targetValueRange", "-tvr", required=False,
                        type=str,
                        help="Undo the [0, 1] mapping that training applied, so the "
                             "prediction comes back in the target's real units. Either a "
                             "path to the target_value_range.json written by hicTraining "
                             "--scaleTargetToUnitRange, or \"min,max\" given directly.")
    parser.add_argument("--rebuildProcesses", "-rp", required=False,
                        type=int, default=0,
                        help="How many chromosomes to rebuild into matrices at once.")
    parser.add_argument("--includeRegions", "-ir", required=False,
                        type=str,
                        nargs='+',
                        help="One or more BED files. Predict ONLY the sliding-window "
                             "positions whose target window lies entirely inside one of "
                             "these regions, instead of running over the whole chromosome.")
    parser.add_argument("--saveMemory", "-sm", action="store_true",
                        help="Enable memory-saving mode for prediction")
    parser.add_argument("--numberOfBatches", "-nb", required=False,
                        type=int,
                        default=20,
                        help="Number of batches to split predictions when --saveMemory is enabled")
    parser.add_argument("--whichGPU", "-wgpu", required=False,
                        type=int,
                        default=1,
                        help="Specify which GPU to use for prediction in the single GPU case.")
    parser.add_argument("--mode", "-m", required=False,
                        type=str,
                        choices=["create-data", "predict", "make-matrix", 'all'],
                        default="all",
                        help="Operation mode: 'create-data' (only create TFRecord data; CPU only), "
                                "'predict' (run prediction using the trained model; requires GPU), "
                                "'make-matrix' (build final matrix from existing predictions; CPU only), "
                                "'all' (run all steps sequentially)")
    parser.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))
    return parser

def createDataContainer(pChromNameList, pOutputFolder, pChromatinFolder, pBinSize, 
                        scalefactors, clampfactors, scalematrix, windowSize, flankingsize, maxdist, batchSize,
                        includedRegions=None, inputMode="random2d", windowMode="diagonal", gridStride=None,
                        chromSizesPath=None, randomSeed=42):
    
    containerCls = dataContainer.DataContainer
    testdataContainerList = []
    chromSizesSource = chromSizesPath or pChromatinFolder
    if inputMode == "random2d" and chromSizesSource is None:
        defaultNpzDir = "data/chrom_npz_homo_sapiens_GRCh38_250kb_tss_count"
        if os.path.exists(defaultNpzDir):
            chromSizesSource = defaultNpzDir
            log.info("[PREDICT-DATA] Using default chromosome sizes source: %s", chromSizesSource)

    for chrom in pChromNameList:
        numBinsChrom = None
        if inputMode == "random2d":
            if chromSizesSource is None:
                raise SystemExit(
                    "Aborting. In 'random2d' mode, please provide --chromSizesPath (-csp) or -tcp pointing to "
                    "chrom_index.csv or an .npz directory so chromosome bin counts can be determined."
                )
            numBinsChrom = utils.getChromBinsFromNpzSource(chromSizesSource, chrom, pBinSize=pBinSize)
            log.info("[PREDICT-DATA] Resolved n_bins=%d for chr%s from %s", numBinsChrom, str(chrom), str(chromSizesSource))
        elif chromSizesPath is not None and os.path.exists(chromSizesPath):
            try:
                numBinsChrom = utils.getChromBinsFromNpzSource(chromSizesPath, chrom, pBinSize=pBinSize)
                log.info("[PREDICT-DATA] Resolved target n_bins=%d for chr%s from %s (bigwig mode)", numBinsChrom, str(chrom), str(chromSizesPath))
            except Exception:
                numBinsChrom = None
        testdataContainerList.append(
            containerCls(
                chromosome=chrom,
                matrixFilePath=None,
                chromatinFolder=pChromatinFolder if inputMode == "bigwig" else None,
                binSize=pBinSize,
                inputMode=inputMode,
                numBins=numBinsChrom,
                randomSeed=randomSeed
            )
        )
    effectiveFlanking = 0 if inputMode == "random2d" else flankingsize
    loadParams = {
        "scaleFeatures": scalefactors,
        "clampFeatures": clampfactors,
        "scaleTargets": scalematrix,
        "windowSize": windowSize,
        "flankingSize": effectiveFlanking,
        "maximumDistance": maxdist,
        "includedRegions": includedRegions,
        "windowMode": windowMode,
        "gridStride": gridStride,
    }
    if len(testdataContainerList) == 0:
        msg = "Exiting. No data found"
        print(msg)
        return
    required_bins = windowSize + 2 * effectiveFlanking
    container0 = None
    keptContainers = []
    tfRecordFilenames = []
    sampleSizeList = []
    windowStartsList = []
    windowCoordsList = []
    matrixSizeList = []
    originalBinsList = []
    flankingSizeList = []
    for container in testdataContainerList:
        container.loadData(**loadParams)
        if not container.hasEnoughBins():
            nr_bins = container.FactorDataArray.shape[0] if container.FactorDataArray is not None else 0
            msg = ("Skipping chr{:s}: {:d} bins at binSize {:s} < required {:d} "
                   "(windowSize {:d} + 2x flanking). Use a finer binSize to include it.").format(
                       str(container.chromosome), nr_bins, str(pBinSize), required_bins, windowSize)
            print(msg)
            log.warning(msg)
            container.unloadData()
            continue
        if container0 is None:
            container0 = container
        if not container0.checkCompatibility(container):
            msg = "Aborting. Incompatible data"
            raise SystemExit(msg)
        tfRecordFilenames.append(container.writeTFRecord(pOutputFolder=pOutputFolder,
                                                        pRecordSize=None)[0])
        sampleSizeList.append(int(np.ceil(container.getNumberSamples() / batchSize)))
        nr_bins_chrom = container.FactorDataArray.shape[0] if container.FactorDataArray is not None else 0
        orig_bins_chrom = container.original_num_bins if container.original_num_bins is not None else nr_bins_chrom
        matrixSizeList.append(int(nr_bins_chrom))
        originalBinsList.append(int(orig_bins_chrom))
        flankingSizeList.append(int(container.flankingSize))
        windowStartsList.append(None if container.sampleIndices is None
                                else np.asarray(container.sampleIndices, dtype=np.int64))
        windowCoordsList.append(container.getWindowCoordinates())
        keptContainers.append(container)

    if container0 is None:
        msg = ("Exiting. No requested chromosome has enough bins at binSize {:s} for windowSize {:d} "
               "(need >= {:d} bins). Predict at a finer binSize (smaller -b).").format(
                   str(pBinSize), windowSize, required_bins)
        print(msg)
        raise SystemExit(msg)

    nr_factors = container0.nr_factors
    for container in keptContainers:
        container.unloadData()
    return (
        keptContainers,
        tfRecordFilenames,
        sampleSizeList,
        nr_factors,
        windowStartsList,
        matrixSizeList,
        originalBinsList,
        flankingSizeList,
        windowCoordsList,
    )

def prediction(pTrainedModel, pPredictionChromosomesFolders, pPredictionChromosomes, pOutputFolder, pMultiplier, pBinSize, pBatchSize, pWindowSize, pMatrixOutputName, pParameterOutputFile, pSaveMemory=False, pNumberOfBatches=20, pScope=None, pMode="all", pIncludeRegions=None, pTargetValueRange=None, pRebuildProcesses=0, pInputMode=None, pWindowMode="diagonal", pGridStride=None, pChromSizesPath=None, pSeed=42, pLogTransformTarget=False, pNpzOutputFolder=None):
    trainedmodel = pTrainedModel
    predictionChromosomesFolders = pPredictionChromosomesFolders
    predictionChromosomes = pPredictionChromosomes
    outputFolder = pOutputFolder
    multiplier = pMultiplier
    binSize = pBinSize if pBinSize is not None else 250000
    batchSize = pBatchSize
    windowSize = pWindowSize

    os.makedirs(outputFolder, exist_ok=True)
    utils.configureLogging(os.path.join(outputFolder, "predict.log"))

    inputMode = pInputMode
    if inputMode is None:
        if not predictionChromosomesFolders or utils.isNpzMatrixSource(predictionChromosomesFolders) or str(predictionChromosomesFolders).endswith(".csv"):
            inputMode = "random2d"
        else:
            inputMode = "bigwig"
    windowMode = pWindowMode or "diagonal"
    logTransformTarget = bool(pLogTransformTarget)

    # Auto-detect target_value_range.json next to trainedModel if not explicitly passed
    if not pTargetValueRange and trainedmodel:
        candidateRange = os.path.join(os.path.dirname(os.path.abspath(trainedmodel)), "target_value_range.json")
        if os.path.exists(candidateRange):
            pTargetValueRange = candidateRange
            log.info("[PREDICT-SCALE] Auto-detected target_value_range.json at %s", candidateRange)
            print(f"[PREDICT-SCALE] Auto-detected target_value_range.json at {candidateRange}", flush=True)

    #resolve the value range the model was trained against, if one was used
    targetValueRange = None
    if pTargetValueRange:
        if os.path.exists(str(pTargetValueRange)):
            lo, hi, log_flag = utils.loadValueRangeMeta(pTargetValueRange)
            targetValueRange = (lo, hi)
            if log_flag:
                logTransformTarget = True
        else:
            try:
                lo, hi = (float(x) for x in str(pTargetValueRange).split(","))
                targetValueRange = (lo, hi)
            except Exception:
                raise SystemExit(
                    "--targetValueRange must be a path to target_value_range.json "
                    "or \"min,max\"; got {!r}".format(pTargetValueRange))
        log.info(
            "[PREDICT-SCALE] Undoing unit-range mapping with [%g, %g] (expm1=%s)",
            targetValueRange[0], targetValueRange[1], str(logTransformTarget)
        )
        print(
            f"[PREDICT-SCALE] Undoing unit-range mapping with [{targetValueRange[0]:.4f}, {targetValueRange[1]:.4f}] "
            f"(expm1={logTransformTarget})",
            flush=True
        )
    scalefactors = True
    clampfactors = False
    scalematrix = True
    maxdist = None
    windowSize = int(windowSize)
    flankingsize = 0 if inputMode == "random2d" else windowSize

    log.info(
        "[PREDICT-SETUP] inputMode=%s, windowMode=%s, gridStride=%s, windowSize=%d, flankingSize=%d, "
        "binSize=%d, mode=%s, seed=%d",
        inputMode, windowMode, str(pGridStride), windowSize, flankingsize, binSize, pMode, pSeed
    )

    paramDict = locals().copy()
    predictionChunksDir = os.path.join(outputFolder, "prediction_chunks")
    matrixChunksDir = os.path.join(outputFolder, "matrix_chunks")
    npzOutDir = pNpzOutputFolder if pNpzOutputFolder else os.path.join(outputFolder, "npz_matrices")
    os.makedirs(predictionChunksDir, exist_ok=True)
    os.makedirs(matrixChunksDir, exist_ok=True)
    os.makedirs(npzOutDir, exist_ok=True)
        
    chromNameList = sorted([x.lstrip("chr") for x in predictionChromosomes])

    if pMode in ["create-data", "all"]:
        (
            testdataContainerList,
            tfRecordFilenames,
            sampleSizeList,
            nr_factors,
            windowStartsList,
            matrixSizeList,
            originalBinsList,
            flankingSizeList,
            windowCoordsList,
        ) = createDataContainer(
            pChromNameList=chromNameList, 
            pOutputFolder=outputFolder, 
            pChromatinFolder=predictionChromosomesFolders, 
            pBinSize=binSize, 
            scalefactors=scalefactors, 
            clampfactors=clampfactors, 
            scalematrix=scalematrix, 
            windowSize=windowSize, 
            flankingsize=flankingsize, 
            maxdist=maxdist, 
            batchSize=batchSize,
            includedRegions=pIncludeRegions,
            inputMode=inputMode,
            windowMode=windowMode,
            gridStride=pGridStride,
            chromSizesPath=pChromSizesPath,
            randomSeed=pSeed,
        )
        if pMode == "create-data":
            save_vars = {
                "testdataContainerList": testdataContainerList,
                "tfRecordFilenames": tfRecordFilenames,
                "sampleSizeList": sampleSizeList,
                "nr_factors": nr_factors,
                "windowStartsList": windowStartsList,
                "matrixSizeList": matrixSizeList,
                "originalBinsList": originalBinsList,
                "flankingSizeList": flankingSizeList,
                "windowCoordsList": windowCoordsList,
                "inputMode": inputMode,
                "windowMode": windowMode,
            }

            pickle_path = os.path.join(outputFolder, "prediction_vars.pkl")
            with open(pickle_path, "wb") as fh:
                pickle.dump(save_vars, fh, protocol=pickle.HIGHEST_PROTOCOL)

            log.info("Wrote prediction variables to %s", pickle_path)
            return
    elif pMode == "predict":
        pickle_path = os.path.join(outputFolder, "prediction_vars.pkl")
        if not os.path.exists(pickle_path):
            log.error("Pickle file with prediction variables not found at %s. Please run with --mode create-data first.", pickle_path)
            return
        
        with open(pickle_path, "rb") as fh:
            loaded_vars = pickle.load(fh)

        testdataContainerList = loaded_vars["testdataContainerList"]
        tfRecordFilenames = loaded_vars["tfRecordFilenames"]
        sampleSizeList = loaded_vars["sampleSizeList"]
        nr_factors = loaded_vars["nr_factors"]
        windowStartsList = loaded_vars.get("windowStartsList")
        matrixSizeList = loaded_vars.get("matrixSizeList")
        originalBinsList = loaded_vars.get("originalBinsList", matrixSizeList)
        flankingSizeList = loaded_vars.get("flankingSizeList", [flankingsize] * len(testdataContainerList))
        windowCoordsList = loaded_vars.get("windowCoordsList", [None] * len(testdataContainerList))
        inputMode = loaded_vars.get("inputMode", inputMode)
        windowMode = loaded_vars.get("windowMode", windowMode)

    if pMode in ["predict", "all"]:
        use_embedding = (inputMode == "bigwig")
        enforce_symmetry = (windowMode == "diagonal")
        if use_embedding and testdataContainerList and isinstance(testdataContainerList[0].storedFeatures, dict):
            sf_dict = testdataContainerList[0].storedFeatures
            if "factorData" in sf_dict and sf_dict["factorData"].get("shape") is not None:
                stored_shape = sf_dict["factorData"]["shape"]
                if len(stored_shape) >= 2:
                    nr_factors = int(stored_shape[1])
        trained_GAN = hicGAN.HiCGAN(
            log_dir=outputFolder,
            number_factors=nr_factors,
            input_size=windowSize,
            scope=pScope,
            use_embedding=use_embedding,
            enforce_symmetry=enforce_symmetry,
        )
        trained_GAN.loadGenerator(trainedModelPath=trainedmodel)
        predictionFiles = []
        predictedChroms = []

        rebuildPool = None
        rebuildAsync = []
        isGridMode = (windowMode == "grid")
        useBand = (not isGridMode) and (pIncludeRegions is None) and bool(matrixSizeList)
        bandStarts = []
        if pMode == "all":
            nRebuild = utils.rebuildProcessCount(pRebuildProcesses, windowSize,
                                                 matrixSizeList, len(tfRecordFilenames))
            log.info("Overlapping rebuild with prediction, %d process(es)", nRebuild)
            print("rebuilding alongside prediction with {:d} process(es)".format(nRebuild),
                  flush=True)
            rebuildPool = multiprocessing.get_context("spawn").Pool(processes=nRebuild)

        for idx, (record, container, nr_samples) in enumerate(
                zip(tfRecordFilenames, testdataContainerList, sampleSizeList)):
            chrom = container.chromosome
            storedFeaturesDict = container.storedFeatures
            actualFlanking = int(flankingSizeList[idx]) if flankingSizeList else flankingsize
            origBins = int(originalBinsList[idx]) if originalBinsList else int(matrixSizeList[idx])
            matrixSize = int(matrixSizeList[idx])

            testDs = tf.data.TFRecordDataset(record,
                                                num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                                compression_type=records.TFRECORD_COMPRESSION or "")
            testDs = testDs.map(lambda x: records.parse_function(x, storedFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            testDs = testDs.batch(batchSize, drop_remainder=False)
            testDs = testDs.prefetch(tf.data.experimental.AUTOTUNE)

            if isGridMode:
                predFile = os.path.join(predictionChunksDir, f"pred_grid_chr{chrom}.npy")
                gridBatches = []
                for batchPred in trained_GAN.predict_stream(test_ds=testDs):
                    gridBatches.append(np.asarray(batchPred, dtype=np.float16))
                if gridBatches:
                    gridStack = np.concatenate(gridBatches, axis=0)
                else:
                    gridStack = np.zeros((0, windowSize, windowSize), dtype=np.float16)
                np.save(predFile, gridStack)
                written = int(gridStack.shape[0])
                del gridBatches, gridStack
                predictionFiles.append(predFile)
                predictedChroms.append(chrom)
                log.info(
                    "[PREDICT-GRID] Predicted %d 2D grid windows for chr%s -> %s (paddedBins=%d, origBins=%d)",
                    written, chrom, predFile, matrixSize, origBins
                )
                if rebuildPool is not None:
                    matrixFile = os.path.join(matrixChunksDir, f"matrix_chr{chrom}.npz")
                    job = (
                        predFile,
                        matrixFile,
                        windowSize,
                        matrixSize,
                        windowCoordsList[idx],
                        tuple(targetValueRange) if targetValueRange is not None else None,
                        multiplier,
                        origBins,
                        logTransformTarget,
                    )
                    rebuildAsync.append(
                        (chrom, matrixFile, rebuildPool.apply_async(
                            utils.rebuildChromosomeFromGrid, (job,))))
                if os.path.exists(record):
                    os.remove(record)
                continue

            if useBand:
                bandFile = os.path.join(predictionChunksDir, f"band_chr{chrom}.npy")
                band, starts = trained_GAN.predict_band(
                    test_ds=testDs,
                    window_size=windowSize,
                    matrix_size=matrixSize,
                    flanking_size=actualFlanking,
                    window_starts=windowStartsList[idx] if windowStartsList else None)
                written = int(starts.size)
                np.save(bandFile, band)
                del band
                predictionFiles.append(bandFile)
                predictedChroms.append(chrom)
                bandStarts.append(starts)
                log.info(
                    "[PREDICT-BAND] Summed %d windows for chr%s into %s (flanking=%d, paddedBins=%d, origBins=%d)",
                    written, chrom, bandFile, actualFlanking, matrixSize, origBins
                )

                if rebuildPool is not None:
                    matrixFile = os.path.join(matrixChunksDir, f"matrix_chr{chrom}.npz")
                    job = (
                        bandFile,
                        matrixFile,
                        windowSize,
                        matrixSize,
                        starts,
                        tuple(targetValueRange) if targetValueRange is not None else None,
                        multiplier,
                        origBins,
                        logTransformTarget,
                    )
                    rebuildAsync.append(
                        (chrom, matrixFile, rebuildPool.apply_async(
                            utils.rebuildChromosomeFromBand, (job,))))
                if os.path.exists(record):
                    os.remove(record)
                continue

            triu_r, triu_c = np.triu_indices(windowSize)
            log.debug("Streaming predictions and extracting triangles...")
            predFile = os.path.join(predictionChunksDir, f"pred_chr{chrom}.npy")

            rawFile = predFile + ".raw"
            written = 0
            with open(rawFile, "wb") as rawHandle:
                for tri in trained_GAN.predict_stream_triu(test_ds=testDs,
                                                           window_size=windowSize):
                    rawHandle.write(np.ascontiguousarray(tri).tobytes())
                    written += tri.shape[0]
                    del tri
            if written == 0:
                log.warning("No predictions produced for chr%s", chrom)
            shape = (written, int(triu_r.size))
            with open(predFile, "wb") as npyHandle:
                np.lib.format.write_array_header_2_0(
                    npyHandle, {"descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
                                "fortran_order": False, "shape": shape})
                with open(rawFile, "rb") as rawHandle:
                    while True:
                        block = rawHandle.read(64 * 1024 * 1024)
                        if not block:
                            break
                        npyHandle.write(block)
            os.remove(rawFile)

            predictionFiles.append(predFile)
            predictedChroms.append(chrom)
            log.info("Wrote %d predictions for chr%s to %s", written, chrom, predFile)

            if rebuildPool is not None:
                matrixFile = os.path.join(matrixChunksDir, f"matrix_chr{chrom}.npz")
                job = (
                    predFile,
                    matrixFile,
                    windowSize,
                    actualFlanking,
                    pSaveMemory,
                    windowStartsList[idx] if windowStartsList else None,
                    matrixSize,
                    tuple(targetValueRange) if targetValueRange is not None else None,
                    multiplier,
                    origBins,
                    logTransformTarget,
                )
                rebuildAsync.append(
                    (chrom, matrixFile, rebuildPool.apply_async(
                        utils.rebuildChromosomeToFile, (job,))))

            if os.path.exists(record):
                os.remove(record)
        log.debug("Prediction on all chromosomes completed.")
        if rebuildPool is not None and not rebuildAsync:
            rebuildPool.terminate()
            rebuildPool.join()
            rebuildPool = None
        chromNameList = predictedChroms

        log.info("Cleaning up temporary files...")
        for tfrecordfile in tfRecordFilenames:
            if os.path.exists(tfrecordfile):
                os.remove(tfrecordfile)

        chunkFormat = "grid" if isGridMode else ("band" if useBand else "windows")
        save_vars = {
            "predictionFiles": predictionFiles,
            "chromNameList": chromNameList,
            "windowSize": windowSize,
            "windowStartsList": windowStartsList,
            "matrixSizeList": matrixSizeList,
            "originalBinsList": originalBinsList,
            "flankingSizeList": flankingSizeList,
            "windowCoordsList": windowCoordsList,
            "chunkFormat": chunkFormat,
            "bandStarts": bandStarts if useBand else None,
            "inputMode": inputMode,
            "windowMode": windowMode,
        }

        pickle_path = os.path.join(outputFolder, "predictions.pkl")
        with open(pickle_path, "wb") as fh:
            pickle.dump(save_vars, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Wrote predictions to %s", pickle_path)

        if pMode == "predict":
            return
    if pMode == "make-matrix":
        pickle_path = os.path.join(outputFolder, "predictions.pkl")
        if not os.path.exists(pickle_path):
            log.error("Pickle file with predictions not found at %s. Please run with --mode predict first.", pickle_path)
            return
        
        with open(pickle_path, "rb") as fh:
            loaded_vars = pickle.load(fh)

        predictionFiles = loaded_vars.get("predictionFiles")
        chromNameList = loaded_vars["chromNameList"]
        windowStartsList = loaded_vars.get("windowStartsList")
        matrixSizeList = loaded_vars.get("matrixSizeList")
        originalBinsList = loaded_vars.get("originalBinsList", matrixSizeList)
        flankingSizeList = loaded_vars.get("flankingSizeList")
        windowCoordsList = loaded_vars.get("windowCoordsList")
        chunkFormat = loaded_vars.get("chunkFormat", "windows")
        bandStarts = loaded_vars.get("bandStarts")
        if "windowSize" in loaded_vars:
            windowSize = int(loaded_vars["windowSize"])

        if predictionFiles is None and "predList" in loaded_vars:
            predictionFiles = []
            for chrom, predArray in zip(chromNameList, loaded_vars["predList"]):
                predFile = os.path.join(predictionChunksDir, f"pred_chr{chrom}.npy")
                np.save(predFile, np.asarray(predArray, dtype=np.float16), allow_pickle=False)
                predictionFiles.append(predFile)
            log.info("Converted legacy in-memory predictions to disk-backed chunks in %s", predictionChunksDir)

    if pMode in ["make-matrix", "all"]:
        if pMode == "all" and rebuildAsync:
            matrixFiles = []
            for chrom, matrixFile, result in rebuildAsync:
                writtenFile, nnz = result.get()
                matrixFiles.append(writtenFile)
                print("  chr{:<3s} rebuilt, {:,d} stored pixels".format(str(chrom), nnz),
                      flush=True)
            rebuildPool.close()
            rebuildPool.join()

            log.info("[NPZ-EXPORT] Writing per-chromosome .npz files and chrom_index.csv to %s...", npzOutDir)
            utils.writeNpzDataset(
                pMatrixList=matrixFiles,
                pChromosomeList=chromNameList,
                pOutputDir=npzOutDir
            )

            matrixname = os.path.join(outputFolder, pMatrixOutputName)
            log.info("Writing predicted matrix to disk on %s..." % matrixname)
            utils.writeCooler(pMatrixList=matrixFiles,
                              pBinSizeInt=binSize,
                              pOutfile=matrixname,
                              pChromosomeList=chromNameList)

            parameterFile = os.path.join(outputFolder, pParameterOutputFile)
            with open(parameterFile, "w") as csvfile:
                dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
                dictWriter.writeheader()
                dictWriter.writerow(paramDict)
            return

        if pMode == "all":
            if "predictionFiles" not in locals():
                predictionFiles = []
                for chrom in chromNameList:
                    predFile = os.path.join(predictionChunksDir, f"pred_chr{chrom}.npy")
                    if os.path.exists(predFile):
                        predictionFiles.append(predFile)

        if not predictionFiles:
            log.error("No prediction chunk files found in %s", predictionChunksDir)
            return

        if not windowStartsList:
            windowStartsList = [None] * len(chromNameList)
        if not matrixSizeList:
            matrixSizeList = [None] * len(chromNameList)
        if not locals().get("originalBinsList"):
            originalBinsList = matrixSizeList
        if not locals().get("flankingSizeList"):
            flankingSizeList = [flankingsize] * len(chromNameList)
        if not locals().get("windowCoordsList"):
            windowCoordsList = [None] * len(chromNameList)

        fmt = locals().get("chunkFormat", "band" if locals().get("useBand") else "windows")
        if fmt == "grid":
            worker = utils.rebuildChromosomeFromGrid
        elif fmt == "band":
            worker = utils.rebuildChromosomeFromBand
        else:
            worker = utils.rebuildChromosomeToFile

        jobs = []
        for i, (chrom, predFile, windowStarts, matrixSize) in enumerate(zip(
                chromNameList, predictionFiles, windowStartsList, matrixSizeList)):
            matrixFile = os.path.join(matrixChunksDir, f"matrix_chr{chrom}.npz")
            origBins = originalBinsList[i] if originalBinsList else matrixSize
            actualFlanking = flankingSizeList[i] if flankingSizeList else flankingsize
            if fmt == "grid":
                jobs.append((
                    predFile,
                    matrixFile,
                    windowSize,
                    matrixSize,
                    windowCoordsList[i],
                    tuple(targetValueRange) if targetValueRange is not None else None,
                    multiplier,
                    origBins,
                    logTransformTarget,
                ))
            elif fmt == "band":
                starts = bandStarts[i] if bandStarts else None
                if starts is None:
                    raise ValueError(
                        "chr{:s} was summed into a band but its window starts are "
                        "missing from predictions.pkl".format(str(chrom)))
                jobs.append((
                    predFile,
                    matrixFile,
                    windowSize,
                    matrixSize,
                    starts,
                    tuple(targetValueRange) if targetValueRange is not None else None,
                    multiplier,
                    origBins,
                    logTransformTarget,
                ))
            else:
                jobs.append((
                    predFile,
                    matrixFile,
                    windowSize,
                    actualFlanking,
                    pSaveMemory,
                    windowStarts,
                    matrixSize,
                    tuple(targetValueRange) if targetValueRange is not None else None,
                    multiplier,
                    origBins,
                    logTransformTarget,
                ))

        nProcesses = utils.rebuildProcessCount(pRebuildProcesses, windowSize,
                                               matrixSizeList, len(jobs))
        log.info("Rebuilding %d chromosomes with %d process(es)", len(jobs), nProcesses)
        print("rebuilding {:d} chromosomes with {:d} process(es)".format(
            len(jobs), nProcesses), flush=True)
        matrixFiles = []
        if nProcesses > 1:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=nProcesses) as pool:
                for chrom, (matrixFile, nnz) in zip(
                        chromNameList, pool.imap(worker, jobs)):
                    matrixFiles.append(matrixFile)
                    print("  chr{:<3s} rebuilt, {:,d} stored pixels".format(str(chrom), nnz),
                          flush=True)
        else:
            for chrom, job in zip(chromNameList, jobs):
                matrixFile, nnz = worker(job)
                matrixFiles.append(matrixFile)
                print("  chr{:<3s} rebuilt, {:,d} stored pixels".format(str(chrom), nnz),
                      flush=True)

        log.info("[NPZ-EXPORT] Writing per-chromosome .npz files and chrom_index.csv to %s...", npzOutDir)
        utils.writeNpzDataset(
            pMatrixList=matrixFiles,
            pChromosomeList=chromNameList,
            pOutputDir=npzOutDir
        )

        matrixname = os.path.join(outputFolder, pMatrixOutputName)
        log.info("Writing predicted matrix to disk on %s..." % matrixname)   

        utils.writeCooler(pMatrixList=matrixFiles, 
                      pBinSizeInt=binSize, 
                      pOutfile=matrixname, 
                      pChromosomeList=chromNameList)

        parameterFile = os.path.join(outputFolder, pParameterOutputFile) 
        with open(parameterFile, "w") as csvfile:
            dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
            dictWriter.writeheader()
            dictWriter.writerow(paramDict)
        
       


def main(args=None):
    args = parse_arguments().parse_args(args)

    if args.mode == "all" or args.mode == "predict":
        physical_gpus = tf.config.list_physical_devices('GPU')

        if physical_gpus:
            try:
                index = args.whichGPU - 1
                if index < 0 or index >= len(physical_gpus):
                    raise ValueError(f"Invalid GPU index: {index}. Available: {len(physical_gpus)}")

                tf.config.set_visible_devices(physical_gpus[index], 'GPU')
                tf.config.experimental.set_memory_growth(physical_gpus[index], True)

                log.info(f"Physically selected GPU: {physical_gpus[index].name}")
                log.info("Other GPUs are now invisible to TensorFlow.")

            except Exception as e:
                print("Error setting up GPU: {}".format(e))
        else:
            log.info("No GPUs found. Running on CPU.")
        
    prediction(pTrainedModel=args.trainedModel,
        pPredictionChromosomesFolders=args.predictionChromosomesFolders,
        pPredictionChromosomes=args.predictionChromosomes,
        pOutputFolder=args.outputFolder,
        pMultiplier=args.multiplier,
        pBinSize=args.binSize,
        pBatchSize=args.batchSize,
        pWindowSize=args.windowSize,
        pMatrixOutputName=args.matrixOutputName,
        pParameterOutputFile=args.parameterOutputFile,
        pSaveMemory=args.saveMemory,
        pNumberOfBatches=args.numberOfBatches,
        pScope=None, 
        pMode=args.mode,
        pIncludeRegions=args.includeRegions,
        pTargetValueRange=args.targetValueRange,
        pRebuildProcesses=args.rebuildProcesses,
        pInputMode=args.inputMode,
        pWindowMode=args.windowMode,
        pGridStride=args.gridStride,
        pChromSizesPath=args.chromSizesPath,
        pSeed=args.seed,
        pLogTransformTarget=args.logTransformTarget,
        pNpzOutputFolder=args.npzOutputFolder)
