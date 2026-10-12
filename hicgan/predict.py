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
    # parser = argparse.ArgumentParser(description="Hi-cGAN Prediction")
    parser.add_argument("--trainedModel", "-trm", required=False,
                        type=str,
                        help="Trained generator model to predict from")
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
                        help="Bin size for binning the chromatin features")
    parser.add_argument("--batchSize", "-bs", required=False,
                        type=int,
                        default=32, 
                        help="Batch size for predicting")
    # [Orijinal Kod - Korundu & Genişletildi]:
    # Orijinal projede minimum pencere boyutu 64 idi:
    # parser.add_argument("--windowSize", "-ws", required=False,
    #                     type=int,
    #                     choices=[64, 128, 256, 512, 768, 1024],
    #                     help="Window size for predicting; must be the same as in trained model. Supported values are 64, 128, and 256")
    # 250 kb çözünürlük için 16 ve 32 pencere boyutları eklendi.
    parser.add_argument("--windowSize", "-ws", required=False,
                        type=int,
                        choices=[16, 32, 64, 128, 256, 512, 768, 1024],
                        help="Window size for predicting; must be the same as in trained model.")
    # [Yeni Argüman - Eklendi]:
    # Eğitimde kullanılan flankingSize (varsayılan 0) ile tutarlı çalışması için eklendi.
    parser.add_argument("--flankingSize", "-fls", required=False,
                        type=int, default=0,
                        help="Flanking size in bins (default: 0). Must match training flanking size.")
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

def prediction(pTrainedModel, pPredictionChromosomesFolders, pPredictionChromosomes, pOutputFolder, pMultiplier, pBinSize, pBatchSize, pWindowSize, pMatrixOutputName, pParameterOutputFile, pSaveMemory=False, pNumberOfBatches=20, pScope=None, pMode="all", pIncludeRegions=None, pTargetValueRange=None, pRebuildProcesses=0, pFlankingSize=0):
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
    # [Orijinal Kod - Korundu & Güncellendi]:
    # Orijinal projede flankingsize doğrudan windowSize olarak atanıyordu:
    # flankingsize = windowSize
    # 2D TSS NPZ matris girdilerinde eğitim parametresiyle tutarlı olmak ve veri kaybını önlemek için:
    flankingsize = pFlankingSize if pFlankingSize is not None else 0

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
        trained_GAN = hicGAN.HiCGAN(log_dir=outputFolder, number_factors=nr_factors, scope=pScope)
        trained_GAN.loadGenerator(trainedModelPath=trainedmodel)
        predictionFiles = []
        predictedChroms = []

        # In all-mode the rebuild of a finished chromosome is started while the
        # next one is still on the GPU. The two phases use different resources
        # -- the forward pass is GPU-bound and the rebuild is CPU-bound -- and
        # chromosomes are independent, so waiting for the last prediction before
        # starting the first rebuild leaves the card idle for the whole rebuild.
        # The pool is the same one make-matrix uses; only the moment of
        # submission differs.
        rebuildPool = None
        rebuildAsync = []
        #Summing on the GPU needs every window position to be predicted, so it
        #is the path for a whole chromosome and not for a restricted set of
        #regions, where the individual windows are the output rather than an
        #intermediate.
        useBand = pIncludeRegions is None and bool(matrixSizeList)
        bandStarts = []
        if pMode == "all":
            nRebuild = utils.rebuildProcessCount(pRebuildProcesses, windowSize,
                                                 matrixSizeList, len(tfRecordFilenames))
            log.info("Overlapping rebuild with prediction, %d process(es)", nRebuild)
            print("rebuilding alongside prediction with {:d} process(es)".format(nRebuild),
                  flush=True)
            rebuildPool = multiprocessing.get_context("spawn").Pool(processes=nRebuild)

        # Derive the chromosome from the container itself rather than the
        # originally requested chromNameList: some chromosomes may have been
        # skipped (too few bins at this binSize), so positional zipping with
        # the full request list would misalign names and predictions.
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

            # The overlapping windows are summed on the GPU, where they are
            # produced, and only the resulting band comes back. Every cell of
            # the rebuilt matrix is covered by up to windowSize windows, so the
            # per-window triangles are that many redundant copies of the same
            # result: at 2 kb a genome is 397 GB of them against 3 GB of band.
            # Writing them out and reading them back was the whole cost of
            # prediction on anything but a local disk.
            #
            # The per-window path is kept for --includeRegions, where the
            # windows are the point rather than an intermediate, and where they
            # are few enough to store.
            if useBand:
                bandFile = os.path.join(predictionChunksDir, f"band_chr{chrom}.npy")
                matrixSize = int(matrixSizeList[idx])
                # [Orijinal Kod - Korundu & Güncellendi]:
                # Orijinal projede flanking_size=windowSize olarak sabit verilmişti:
                # flanking_size=windowSize,
                # 2D TSS NPZ matris girdilerinde flanking=0 olduğunda bu durum boyut aşımına
                # (window at ... exceeds ... bins) yol açıyordu. flankingsize parametresi bağlandı.
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

            # Stream predictions one batch at a time. Each batch is reduced to
            # its upper-triangle as float16 immediately, so the full-chromosome
            # float32 stack is never held in memory and the dataset is decoded
            # exactly once (the previous save-memory path used dataset.skip(),
            # which re-decodes the dataset O(numberOfBatches) times).
            triu_r, triu_c = np.triu_indices(windowSize)
            log.debug("Streaming predictions and extracting triangles...")
            predFile = os.path.join(predictionChunksDir, f"pred_chr{chrom}.npy")

            # Each batch is written to disk as it arrives. The previous version
            # appended every batch to a list and then called np.concatenate,
            # holding the whole chromosome twice: once as the list, once as the
            # joined copy. At a bin size of 2048 chromosome 1 is roughly 32 GB of
            # float16, so the peak was 64 GB and the out-of-memory handler killed
            # the run. Writing as we go bounds the peak at a single batch.
            #
            # The rows are streamed to a raw file and the .npy header is written
            # afterwards, once the true row count is known. Sizing a memory-mapped
            # array up front would need that count in advance, and the estimate
            # taken from the data container is not always exact.
            rawFile = predFile + ".raw"
            written = 0
            with open(rawFile, "wb") as rawHandle:
                # the upper triangle is extracted inside the graph, so only the
                # reduced float16 array crosses the bus and the host does no
                # gather between two GPU calls
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
                # [Orijinal Kod - Korundu & Güncellendi]:
                # job = (predFile, matrixFile, windowSize, windowSize, pSaveMemory, ...)
                # 4. parametre olan flankingSize için flankingsize değişkeni verilir.
                job = (predFile, matrixFile, windowSize, flankingsize, pSaveMemory,
                       windowStartsList[idx] if windowStartsList else None,
                       matrixSizeList[idx] if matrixSizeList else None,
                       tuple(targetValueRange) if targetValueRange is not None else None,
                       multiplier)
                rebuildAsync.append(
                    (chrom, matrixFile, rebuildPool.apply_async(
                        utils.rebuildChromosomeToFile, (job,))))

            # the tfrecord is of no further use once its chromosome is predicted,
            # and at a small bin size it is the largest thing on the scratch disk
            if os.path.exists(record):
                os.remove(record)
        log.debug("Prediction on all chromosomes completed.")
        if rebuildPool is not None and not rebuildAsync:
            rebuildPool.terminate()
            rebuildPool.join()
            rebuildPool = None
        # keep the downstream chromosome list consistent with what was predicted
        chromNameList = predictedChroms

        log.info("Cleaning up temporary files...")
        for tfrecordfile in tfRecordFilenames:
            if os.path.exists(tfrecordfile):
                os.remove(tfrecordfile)
        #Written in "all" mode as well as in "predict" mode. The GPU pass is the
        #part that cannot be repeated cheaply, and without this file a rebuild
        #that fails, or one you want to redo with different settings, costs the
        #whole prediction again: the triangles on disk are useless without the
        #window positions they came from. Sixty kB buys that back.
        save_vars = {
            "predictionFiles": predictionFiles,
            "chromNameList": chromNameList,
            "windowSize": windowSize,
            "flankingSize": flankingsize,
            "windowStartsList": windowStartsList,
            "matrixSizeList": matrixSizeList,
            #"band" means the files hold the summed band and the window starts
            #it was summed from; "windows" means one triangle per window, which
            #still has to be folded. make-matrix cannot tell them apart from the
            #arrays alone.
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
        #absent in pickles written before the band was summed on the GPU, which
        #could only have held per-window triangles
        chunkFormat = loaded_vars.get("chunkFormat", "windows")
        bandStarts = loaded_vars.get("bandStarts")
        if "windowSize" in loaded_vars:
            windowSize = int(loaded_vars["windowSize"])
        if "flankingSize" in loaded_vars:
            flankingsize = int(loaded_vars["flankingSize"])

        # Backward compatibility: older pickles stored all predictions in-memory.
        if predictionFiles is None and "predList" in loaded_vars:
            predictionFiles = []
            for chrom, predArray in zip(chromNameList, loaded_vars["predList"]):
                predFile = os.path.join(predictionChunksDir, f"pred_chr{chrom}.npy")
                np.save(predFile, np.asarray(predArray, dtype=np.float16), allow_pickle=False)
                predictionFiles.append(predFile)
            log.info("Converted legacy in-memory predictions to disk-backed chunks in %s", predictionChunksDir)

    if pMode in ["make-matrix", "all"]:
        # all-mode already submitted every chromosome during the prediction loop.
        # Most of them have finished by now; this only waits for the tail.
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

            parameterFile = os.path.join(outputFolder, pParameterOutputFile)
            with open(parameterFile, "w") as csvfile:
                dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
                dictWriter.writeheader()
                dictWriter.writerow(paramDict)
            return

        if pMode == "all":
            # In all-mode the prediction step produced the chunk file list.
            if "predictionFiles" not in locals():
                predictionFiles = []
                for chrom in chromNameList:
                    predFile = os.path.join(predictionChunksDir, f"pred_chr{chrom}.npy")
                    if os.path.exists(predFile):
                        predictionFiles.append(predFile)

        if not predictionFiles:
            log.error("No prediction chunk files found in %s", predictionChunksDir)
            return

        #None when the whole chromosome was predicted, which is the common case
        #and the behaviour rebuildMatrix defaults to
        if not windowStartsList:
            windowStartsList = [None] * len(chromNameList)
        if not matrixSizeList:
            matrixSizeList = [None] * len(chromNameList)

        #Chromosomes are independent of each other: each one reads its own
        #prediction chunk and writes its own matrix chunk, and nothing is shared.
        #So they are rebuilt in parallel, the way hicBuildMatrix parallelises
        #over regions. The worker is in hicgan.lib.utils rather than here so a
        #spawned process does not import TensorFlow.
        #
        #targetValueRange is applied inside the worker, which is why it must be
        #a plain tuple rather than anything holding a file handle.
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
                # [Orijinal Kod - Korundu & Güncellendi]:
                # jobs.append((predFile, matrixFile, windowSize, windowSize, pSaveMemory, ...))
                # 4. parametre olan flankingSize için flankingsize değişkeni verilir.
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

        parameterFile = os.path.join(outputFolder, pParameterOutputFile) 
        with open(parameterFile, "w") as csvfile:
            dictWriter = csv.DictWriter(csvfile, fieldnames=sorted(list(paramDict.keys())))
            dictWriter.writeheader()
            dictWriter.writerow(paramDict)
        
       


def main(args=None):
    args = parse_arguments().parse_args(args)
    # print(args)
    # 1. Get all physical GPUs

    if args.mode == "all" or args.mode == "predict":
        physical_gpus = tf.config.list_physical_devices('GPU')

        if physical_gpus:
            try:
                # Calculate index (assuming args.whichGPU is 1-based data like 1, 2, 3...)
                index = args.whichGPU - 1
                if index < 0 or index >= len(physical_gpus):
                    raise ValueError(f"Invalid GPU index: {index}. Available: {len(physical_gpus)}")

                # 2. Key Step: Make ONLY the selected GPU visible to TensorFlow
                # This prevents TF from touching the memory of the other GPU.
                tf.config.set_visible_devices(physical_gpus[index], 'GPU')

                # 3. Set memory growth only on the visible device
                # Note: After setting visibility, we interact with the specific physical object
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
        pFlankingSize=args.flankingSize)
