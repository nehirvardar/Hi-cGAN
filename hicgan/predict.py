import argparse
import multiprocessing
import numpy as np
import os
import csv
import tensorflow as tf
from scipy import sparse
import logging
try:
    from .lib import dataContainer
    from .lib import records
    from .lib import hicGAN
    from .lib import utils
    from hicgan._version import __version__
except (ImportError, ValueError):
    from lib import dataContainer
    from lib import records
    from lib import hicGAN
    from lib import utils
    __version__ = "1.0.0"
import pickle

log = logging.getLogger(__name__)

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser(description="Hi-cGAN Prediction")
    parser.add_argument("--trainedModel", "-trm", required=False,
                        type=str,
                        help="Trained denoiser model to predict from")
    parser.add_argument("--predictionChromosomesFolders", "-tcp", required=False,
                        type=str,
                        help="Path where test data (bigwig files) resides")
    parser.add_argument("--predictionChromosomes", "-pc", required=False,
                        type=str,
                        nargs='+',
                        help="Chromosomes the Hi-C matrix should be predicted. Must be available in all bigwig files")
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
                        help="Output path for predicted cool-file")
    parser.add_argument("--multiplier", "-mul", required=False,
                        type=int, 
                        default=10, 
                        help="Multiplier for scaling the predicted coolers")
    parser.add_argument("--binSize", "-b", required=False,
                        type=int,
                        default=250000,
                        help="Bin size for matrices (default: 250000 for 250kb RNA matrices)")
    parser.add_argument("--batchSize", "-bs", required=False,
                        type=int,
                        default=32, 
                        help="Batch size for predicting")
    parser.add_argument("--windowSize", "-ws", required=False,
                        type=int,
                        default=64,
                        choices=[64, 128, 256, 512, 768, 1024],
                        help="Window size for predicting; must be the same as in trained model. Supported values are 64, 128, and 256")
    parser.add_argument("--dataDir", "-d", required=False,
                        type=str,
                        help="Data directory containing rna_gt (or rna) and random_input folders")
    parser.add_argument("--splitFile", "-sf", required=False,
                        type=str,
                        help="Split file (splits.txt) specifying train/val/test chromosomes")
    parser.add_argument("--split", required=False,
                        type=str,
                        default="test",
                        choices=["test", "train", "val", "all"],
                        help="Which split from splitFile to evaluate (default: test)")
    parser.add_argument("--targetValueRange", "-tvr", required=False,
                        type=str,
                        help="Undo the [0, 1] mapping that training applied, so the "
                             "prediction comes back in the target's real units. Either a "
                             "path to the target_value_range.json written by hicTraining "
                             "--scaleTargetToUnitRange, or \"min,max\" given directly. When "
                             "set, the min-max rescaling and --multiplier are NOT applied: "
                             "those map onto the single brightest pixel of each chromosome, "
                             "which makes values incomparable between chromosomes and "
                             "between runs. If the model was trained without "
                             "--scaleTargetToUnitRange, leave this unset.")
    parser.add_argument("--rebuildProcesses", "-rp", required=False,
                        type=int, default=0,
                        help="How many chromosomes to rebuild into matrices at once. "
                             "Chromosomes are independent, so this is a straight "
                             "speed-up. 0 (the default) picks a number from the free "
                             "memory and the core count, allowing roughly 2.5 x "
                             "windowSize x chromosomeBins x 4 bytes per process. Set it "
                             "explicitly under a scheduler with its own memory limit, or "
                             "to 1 to rebuild one chromosome at a time.")
    parser.add_argument("--includeRegions", "-ir", required=False,
                        type=str,
                        nargs='+',
                        help="One or more BED files. Predict ONLY the sliding-window "
                             "positions whose target window lies entirely inside one of "
                             "these regions, instead of running over the whole chromosome. "
                             "Pass the same BED that training was given as --excludeRegions "
                             "to predict exactly the held-out loci: containment here is the "
                             "exact complement of the overlap test used there, so no "
                             "predicted window can contain a bin the model was trained on. "
                             "The output cooler covers the whole chromosome and is empty "
                             "outside the predicted windows.")
    parser.add_argument("--saveMemory", "-sm", action="store_true",
                        help="Enable memory-saving mode for prediction")
    parser.add_argument("--numberOfBatches", "-nb", required=False,
                        type=int,
                        default=20,
                        help="Number of batches to split predictions when --saveMemory is enabled")
    parser.add_argument("--whichGPU", "-wgpu", required=False,
                        type=int,
                        default=1,
                        help="Specify which GPU to use for prediction in the single GPU case. "
                             "One-based: 1 is the first GPU. The default was the string \"\", "
                             "which argparse ran through type=int and rejected, so omitting this "
                             "argument aborted the run.")
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
                        includedRegions=None):
    
    containerCls = dataContainer.DataContainer
    testdataContainerList = []
    for chrom in pChromNameList:
        testdataContainerList.append(containerCls(chromosome=chrom,
                                                  matrixFilePath=None,
                                                  chromatinFolder=pChromatinFolder,
                                                  binSize=pBinSize)) 
    loadParams = {"scaleFeatures": scalefactors,
                  "clampFeatures": clampfactors,
                  "scaleTargets": scalematrix,
                  "windowSize": windowSize,
                  "flankingSize": flankingsize,
                  "maximumDistance": maxdist,
                  "includedRegions": includedRegions}
    if len(testdataContainerList) == 0:
        msg = "Exiting. No data found"
        print(msg)
        return
    # A sliding window spans windowSize + 2*flankingSize bins. At a coarse
    # binSize a chromosome may have fewer bins than that and would otherwise
    # crash deep inside getNumberSamples. Skip such chromosomes with a clear
    # warning instead of aborting the whole prediction run, so that
    # resolution-independent prediction works for whichever chromosomes fit.
    required_bins = windowSize + 2 * flankingsize
    container0 = None
    keptContainers = []
    tfRecordFilenames = []
    sampleSizeList = []
    # Which sliding-window position each sample came from, and how many bins the
    # chromosome has. Both are needed to put the predicted triangles back in the
    # right place when --includeRegions restricts the sample set, and both are
    # gone after unloadData(), so they are captured here.
    windowStartsList = []
    matrixSizeList = []
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
        matrixSizeList.append(int(nr_bins_chrom))
        windowStartsList.append(None if container.sampleIndices is None
                                else np.asarray(container.sampleIndices, dtype=np.int64))
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
    return keptContainers, tfRecordFilenames, sampleSizeList, nr_factors, windowStartsList, matrixSizeList

def prediction(pTrainedModel, pPredictionChromosomesFolders, pPredictionChromosomes, pOutputFolder, pMultiplier, pBinSize, pBatchSize, pWindowSize, pMatrixOutputName, pParameterOutputFile, pSaveMemory=False, pNumberOfBatches=20, pScope=None, pMode="all", pIncludeRegions=None, pTargetValueRange=None, pRebuildProcesses=0):
    trainedmodel = pTrainedModel
    predictionChromosomesFolders = pPredictionChromosomesFolders
    predictionChromosomes = pPredictionChromosomes
    outputFolder = pOutputFolder
    multiplier = pMultiplier
    binSize = pBinSize
    batchSize = pBatchSize
    windowSize = pWindowSize

    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    #resolve the value range the model was trained against, if one was used
    targetValueRange = None
    if pTargetValueRange:
        if os.path.exists(str(pTargetValueRange)):
            targetValueRange = utils.loadValueRange(pTargetValueRange)
        else:
            try:
                lo, hi = (float(x) for x in str(pTargetValueRange).split(","))
                targetValueRange = (lo, hi)
            except Exception:
                raise SystemExit(
                    "--targetValueRange must be a path to target_value_range.json "
                    "or \"min,max\"; got {!r}".format(pTargetValueRange))
        log.info("undoing the unit-range mapping with [%g, %g]", *targetValueRange)
    scalefactors = True
    clampfactors = False
    scalematrix = True
    maxdist = None
    windowSize = int(windowSize)
    # Set flankingsize to 0 for 250kb 2D RNA matrices
    flankingsize = 0

    paramDict = locals().copy()
    predictionChunksDir = os.path.join(outputFolder, "prediction_chunks")
    matrixChunksDir = os.path.join(outputFolder, "matrix_chunks")
    if not os.path.exists(predictionChunksDir):
        os.makedirs(predictionChunksDir)
    if not os.path.exists(matrixChunksDir):
        os.makedirs(matrixChunksDir)
        
    chromNameList = sorted([x.lstrip("chr") for x in predictionChromosomes])

    if pMode in ["create-data", "all"]:
        testdataContainerList, tfRecordFilenames, sampleSizeList, nr_factors, windowStartsList, matrixSizeList = createDataContainer(
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
            includedRegions=pIncludeRegions)
        if pMode == "create-data":
            save_vars = {
                "testdataContainerList": testdataContainerList,
                "tfRecordFilenames": tfRecordFilenames,
                "sampleSizeList": sampleSizeList,
                "nr_factors": nr_factors,
                "windowStartsList": windowStartsList,
                "matrixSizeList": matrixSizeList,
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
        #absent in pickles written before --includeRegions existed
        windowStartsList = loaded_vars.get("windowStartsList")
        matrixSizeList = loaded_vars.get("matrixSizeList")

    if pMode in ["predict", "all"]:
        trained_GAN = hicGAN.HiCGAN(log_dir=outputFolder, number_factors=nr_factors, input_size=windowSize, scope=pScope)
        trained_GAN.load_denoiser(trainedModelPath=trainedmodel)
        predictionFiles = []
        predictedChroms = []

        rebuildPool = None
        rebuildAsync = []
        useBand = pIncludeRegions is None and bool(matrixSizeList)
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
            testDs = tf.data.TFRecordDataset(record,
                                                num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                                compression_type=records.TFRECORD_COMPRESSION or "")
            testDs = testDs.map(lambda x: records.parse_function(x, storedFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            testDs = testDs.batch(batchSize, drop_remainder=False)
            testDs = testDs.prefetch(tf.data.experimental.AUTOTUNE)

            if useBand:
                bandFile = os.path.join(predictionChunksDir, f"band_chr{chrom}.npy")
                matrixSize = int(matrixSizeList[idx])
                band, starts = trained_GAN.predict_band(
                    test_ds=testDs,
                    window_size=windowSize,
                    matrix_size=matrixSize,
                    flanking_size=flankingsize,
                    window_starts=windowStartsList[idx] if windowStartsList else None)
                written = int(starts.size)
                np.save(bandFile, band)
                del band
                predictionFiles.append(bandFile)
                predictedChroms.append(chrom)
                bandStarts.append(starts)
                log.info("Summed %d windows for chr%s into %s", written, chrom, bandFile)

                if rebuildPool is not None:
                    matrixFile = os.path.join(matrixChunksDir, f"matrix_chr{chrom}.npz")
                    job = (bandFile, matrixFile, windowSize, matrixSize, starts,
                           tuple(targetValueRange) if targetValueRange is not None else None,
                           multiplier)
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
                job = (predFile, matrixFile, windowSize, flankingsize, pSaveMemory,
                       windowStartsList[idx] if windowStartsList else None,
                       matrixSizeList[idx] if matrixSizeList else None,
                       tuple(targetValueRange) if targetValueRange is not None else None,
                       multiplier)
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
        save_vars = {
            "predictionFiles": predictionFiles,
            "chromNameList": chromNameList,
            "windowSize": windowSize,
            "windowStartsList": windowStartsList,
            "matrixSizeList": matrixSizeList,
            "chunkFormat": "band" if useBand else "windows",
            "bandStarts": bandStarts if useBand else None,
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

            matrixname = os.path.join(outputFolder, pMatrixOutputName)
            log.info("Writing predicted matrix to disk on %s..." % matrixname)
            utils.writeCooler(pMatrixList=matrixFiles,
                              pBinSizeInt=binSize,
                              pOutfile=matrixname,
                              pChromosomeList=chromNameList)

            # Save visual evaluation plots comparing Random Input, Ground Truth RNA, and Predicted RNA
            save_visual_results(chromNameList, matrixFiles, predictionChromosomesFolders, outputFolder, windowSize)

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

        bandMode = (locals().get("chunkFormat", "band" if locals().get("useBand") else "windows")
                    == "band")
        worker = utils.rebuildChromosomeFromBand if bandMode else utils.rebuildChromosomeToFile
        jobs = []
        for i, (chrom, predFile, windowStarts, matrixSize) in enumerate(zip(
                chromNameList, predictionFiles, windowStartsList, matrixSizeList)):
            matrixFile = os.path.join(matrixChunksDir, f"matrix_chr{chrom}.npz")
            if bandMode:
                starts = bandStarts[i] if bandStarts else None
                if starts is None:
                    raise ValueError(
                        "chr{:s} was summed into a band but its window starts are "
                        "missing from predictions.pkl".format(str(chrom)))
                jobs.append((predFile, matrixFile, windowSize, matrixSize, starts,
                             tuple(targetValueRange) if targetValueRange is not None else None,
                             multiplier))
            else:
                jobs.append((predFile, matrixFile, windowSize, flankingsize, pSaveMemory,
                             windowStarts, matrixSize,
                             tuple(targetValueRange) if targetValueRange is not None else None,
                             multiplier))

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

        matrixname = os.path.join(outputFolder, pMatrixOutputName)
        log.info("Writing predicted matrix to disk on %s..." % matrixname)   

        utils.writeCooler(pMatrixList=matrixFiles, 
                      pBinSizeInt=binSize, 
                      pOutfile=matrixname, 
                      pChromosomeList=chromNameList)

        # Save visual evaluation plots comparing Random Input, Ground Truth RNA, and Predicted RNA
        save_visual_results(chromNameList, matrixFiles, predictionChromosomesFolders, outputFolder, windowSize)

        parameterFile = os.path.join(outputFolder, pParameterOutputFile) 
        with open(parameterFile, "w") as csvfile:
            dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
            dictWriter.writeheader()
            dictWriter.writerow(paramDict)


def save_visual_results(chrom_list, matrix_files, chromatin_folder, output_folder, window_size, num_windows=3):
    """Generate visual plots comparing the chromosome's Random Input, Ground Truth RNA, and Predicted RNA."""
    if not chromatin_folder:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parent_dir = os.path.dirname(os.path.normpath(chromatin_folder))
    random_dir = os.path.join(parent_dir, "random_input")

    for chrom, mat_file in zip(chrom_list, matrix_files):
        c_clean = str(chrom).replace("chr", "")
        gt_path = os.path.join(chromatin_folder, f"chr{c_clean}.npz")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(chromatin_folder, f"{c_clean}.npz")
        if not os.path.exists(gt_path) or not os.path.exists(mat_file):
            continue

        gt_mat = sparse.load_npz(gt_path).toarray().astype(np.float32)
        if gt_mat.max() > 0:
            gt_mat = gt_mat / gt_mat.max()

        pred_sparse = sparse.load_npz(mat_file).toarray().astype(np.float32)
        # Symmetrize the rebuilt upper-triangular matrix for display and metric comparison
        pred_mat = pred_sparse + pred_sparse.T - np.diag(np.diag(pred_sparse))
        if pred_mat.max() > 0:
            pred_mat = pred_mat / pred_mat.max()

        rand_path = os.path.join(random_dir, f"chr{c_clean}.npy")
        if os.path.exists(rand_path):
            rand_mat = np.load(rand_path).astype(np.float32)
        else:
            rand_mat = np.zeros_like(gt_mat)

        # Select top windows with highest Ground Truth RNA interaction signal
        n_bins = gt_mat.shape[0]
        max_start = max(1, n_bins - window_size)
        stride = max(1, window_size // 2)
        scored_windows = []
        for s in range(0, max_start, stride):
            score = float(np.sum(np.abs(gt_mat[s:s + window_size, s:s + window_size])))
            scored_windows.append((score, s))
        scored_windows.sort(reverse=True, key=lambda x: x[0])

        for w_idx, (_, s) in enumerate(scored_windows[:num_windows], start=1):
            e = s + window_size
            r_win = rand_mat[s:e, s:e]
            g_win = gt_mat[s:e, s:e]
            p_win = pred_mat[s:e, s:e]

            mse = float(np.mean((g_win - p_win) ** 2))
            mask = (g_win > 0) | (p_win > 0)
            if np.sum(mask) > 10 and np.std(g_win[mask]) > 1e-8 and np.std(p_win[mask]) > 1e-8:
                corr = float(np.corrcoef(g_win[mask].ravel(), p_win[mask].ravel())[0, 1])
            else:
                corr = 0.0

            fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
            panels = [
                (r_win, f"Random Input (chr{c_clean})\nBins {s}-{e}"),
                (g_win, f"Ground Truth RNA (chr{c_clean})\nBins {s}-{e}"),
                (p_win, f"Predicted RNA\nMSE: {mse:.4f} | Pearson: {corr:.4f}")
            ]
            for ax, (img, title) in zip(axs, panels):
                im = ax.imshow(img, cmap="coolwarm")
                ax.set_title(title)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig_path = os.path.join(output_folder, f"pred_chr{c_clean}_win_{w_idx}_bin_{s}.png")
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"  Saved visual plot: {fig_path} (MSE: {mse:.4f}, Pearson: {corr:.4f})")


def main(args=None):
    args = parse_arguments().parse_args(args)

    # Resolve chromosomes and RNA folder from --dataDir and --splitFile if provided
    if args.dataDir and not args.predictionChromosomesFolders:
        rna_subdir = "rna_gt" if os.path.isdir(os.path.join(args.dataDir, "rna_gt")) else "rna"
        args.predictionChromosomesFolders = os.path.join(args.dataDir, rna_subdir)

    if args.splitFile and not args.predictionChromosomes:
        selected_chroms = []
        target_split = args.split.lower()
        with open(args.splitFile, "r") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 2:
                    chrom, sp_name = parts[0], parts[1].lower()
                    if target_split == "all" or sp_name == target_split:
                        selected_chroms.append(chrom)
        args.predictionChromosomes = selected_chroms
        print(f"Selected {len(selected_chroms)} '{args.split}' chromosome(s) from {args.splitFile}: {selected_chroms}")

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
        pRebuildProcesses=args.rebuildProcesses)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
