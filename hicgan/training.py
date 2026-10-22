import csv
import os
import numpy as np
import tensorflow as tf
import concurrent.futures
import argparse
from datetime import datetime
import cooler
import h5py

from .lib import hicGAN
from .lib import dataContainer
from .lib import records
from .lib import utils

from hicgan._version import __version__

import logging
import pickle
log = logging.getLogger(__name__)


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainingMatrices", "-tm", required=True,
                        type=str, nargs='+',
                        help="Target matrices for training (.cool files, .npz files, or folders of per-chromosome .npz files).")
    parser.add_argument("--trainingChromosomes", "-tchroms", required=True,
                        type=str, nargs='+',
                        help="Train chromosomes. Must be present in all train matrices.")
    parser.add_argument("--trainingChromatinFolders", "-tcp", required=False,
                        default=None, type=str, nargs='+',
                        help="Path where chromatin factors for training reside (bigwig files). Optional in 'random2d' inputMode.")
    parser.add_argument("--validationMatrices", "-vm", required=True,
                        type=str, nargs='+',
                        help="Target matrices for validation (.cool files, .npz files, or folders of per-chromosome .npz files).")
    parser.add_argument("--validationChromosomes", "-vchroms", required=True,
                        type=str, nargs='+',
                        help="Validation chromosomes. Must be present in all validation matrices.")
    parser.add_argument("--validationChromatinFolders", "-vcp", required=False,
                        default=None, type=str, nargs='+',
                        help="Path where chromatin factors for validation reside (bigwig files). Optional in 'random2d' inputMode.")
    parser.add_argument("--inputMode", "-im", required=False,
                        type=str, choices=["bigwig", "random2d"], default=None,
                        help="Input feature mode: 'bigwig' (1D chromatin BigWig tracks + cnn_embedding) or "
                             "'random2d' (direct 2D symmetric random matrix input, bypassing cnn_embedding). "
                             "Defaults to 'random2d' when -tcp is omitted, otherwise 'bigwig'.")
    parser.add_argument("--windowMode", "-wm", required=False,
                        type=str, choices=["diagonal", "grid"], default="diagonal",
                        help="Windowing mode: 'diagonal' (sliding windows along main diagonal [i:i+W, i:i+W]) or "
                             "'grid' (2D grid over upper triangle [i:i+W, j:j+W] with i <= j to cover off-diagonal contacts).")
    parser.add_argument("--gridStride", "-gs", required=False,
                        type=int, default=None,
                        help="Stride in bins for 'grid' windowMode (defaults to windowSize // 2).")
    parser.add_argument("--binSize", "-b", required=False,
                        type=int, default=250000,
                        help="Bin size in bp when loading .npz matrices that do not store bin size internally (default: 250000).")
    parser.add_argument("--logTransformTarget", "-ltt", required=False,
                        action="store_true", default=False,
                        help="Apply log1p(x) transform to target matrices before [0, 1] scaling (inverted via expm1 in hicPredict).")
    parser.add_argument("--windowSize", "-ws", required=True,
                        type=int, choices=[64, 128, 256, 512, 768, 1024],
                        help="window size for submatrices.")
    parser.add_argument("--outputFolder", "-o", required=True,
                        type=str,
                        help="Folder where trained model and diverse outputs will be stored.")
    parser.add_argument("--epochs", "-ep", required=True,
                        type=int, default=10,
                        help="Number of epochs for training.")
    parser.add_argument("--batchSize", "-bs", required=False,
                        type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--lossWeightPixel", "-lwp", required=False,
                        type=float,
                        default=100.0,
                        help="Loss weight for L1/L2 error of generator.")
    parser.add_argument("--lossWeightDiscriminator", "-lwd", required=False,
                        type=float,
                        default=0.5,
                        help="Loss weight (multiplicator) for the discriminator loss.")
    parser.add_argument("--lossTypePixel", "-ltp", required=False,
                        type=str, choices=["L1", "L2"],
                        default="L1",
                        help="Type of per-pixel loss to use for the generator.")
    parser.add_argument("--lossWeightTV", "-lwt", required=False,
                        type=float,
                        default=1e-10,
                        help="Loss weight for Total-Variation-loss of generator.")
    parser.add_argument("--lossWeightAdversarial", "-lwa", required=False,
                        type=float,
                        default=1.0,
                        help="Loss weight for adversarial loss in generator.")
    parser.add_argument("--learningRateGenerator", "-lrg", required=False,
                        type=float,
                        default=2e-5,
                        help="Learning rate for Adam optimizer of generator.")
    parser.add_argument("--learningRateDiscriminator", "-lrd", required=False,
                        type=float,
                        default=1e-6,
                        help="Learning rate for Adam optimizer of discriminator.")
    parser.add_argument("--beta1", "-b1", required=False,
                        type=float,
                        default=0.5,
                        help="Beta1 parameter for Adam optimizers (gen. and disc.)")
    parser.add_argument("--flipSamples", "-fs", required=False,
                        action='store_true',
                        help="Flip training matrices and chromatin features (data augmentation).")
    parser.add_argument("--noScaleNorm", "-nsn", required=False,
                        action='store_false',
                        help="Do not scale normalization of chromatin features to 0-1 range.")
    parser.add_argument("--scaleTargetToUnitRange", "-stur", required=False,
                        action="store_true",
                        help="Map the target matrices onto [0, 1] using the value range "
                             "MEASURED from the training targets themselves, and write that "
                             "range to target_value_range.json in the output folder so "
                             "hicPredict can invert it. Use this whenever the target can be "
                             "negative -- an Akita-style log observed/expected target is "
                             "about half negative, and the generator ends in a sigmoid, so "
                             "without this the model is asked for values it cannot emit and "
                             "the discriminator can tell real from generated by sign alone.")
    parser.add_argument("--targetValueRange", "-tvr", required=False, type=str,
                        help="Map the target onto [0, 1] at READ time, for TFRecords that "
                             "were written WITHOUT the mapping (i.e. with --noScaleNorm). "
                             "Takes a path to target_value_range.json, \"min,max\", or "
                             "\"auto\" to measure the range from the training matrices and "
                             "write the json. Use this to reuse existing records instead of "
                             "regenerating them: the mapping is a single affine step and "
                             "the expensive parts of a record -- window selection, "
                             "--excludeRegions, --minTargetCoverage -- are already correct "
                             "inside it. Mutually exclusive with --scaleTargetToUnitRange, "
                             "which does the same thing at WRITE time for new records. "
                             "NOTE a negative minimum must be passed with an equals sign, "
                             "--targetValueRange=-2,1.99 , or argparse reads the leading "
                             "minus as another option.")
    parser.add_argument("--validationSteps", "-vs", required=False,
                        type=int, default=0,
                        help="Evaluate only this many validation batches per epoch "
                             "(0 = all, the default). Validation is not distributed "
                             "across GPUs, so its cost is fixed while the training half "
                             "speeds up with every replica: on two GPUs it already takes "
                             "longer than the training itself. Set a few hundred batches "
                             "for multi-GPU runs.")
    parser.add_argument("--minTargetCoverage", "-mtc", required=False,
                        type=float,
                        default=0.0,
                        help="Skip holes in the target matrix: drop every training sample whose "
                             "target submatrix has a smaller fraction of covered bins than this. "
                             "Covered means the bin has at least one contact, so gaps of the Hi-C "
                             "matrix (unmappable regions, or regions a published dataset does not "
                             "cover) are not handed to the model as blocks of zeros. 0 (the default) "
                             "keeps every sample and reproduces the previous behaviour; 0.95 keeps "
                             "only near-fully covered windows.")
    parser.add_argument("--excludeRegions", "-exr", required=False,
                        type=str, nargs="+", default=None,
                        help="BED file(s) listing regions to leave out of training. Every sample "
                             "whose target window overlaps one of these regions is dropped. Use it "
                             "to hold regions out deliberately, e.g. another method's test set or "
                             "a blacklist. Chromosome names may be given with or without the 'chr' "
                             "prefix. The chromatin features of a kept sample may still reach into "
                             "an excluded region through its flanks; no target from an excluded "
                             "region is ever used.")
    parser.add_argument("--figureFileFormat", "-ft", required=False,
                        type=str, choices=["png", "pdf", "svg"],
                        default="png",
                        help="Figure type for all plots.")
    parser.add_argument("--recordSize", "-rs", required=False,
                        type=int,
                        default=2000,
                        help="Approx. size (number of samples) of the tfRecords used in the data pipeline for training.")
    parser.add_argument("--plotFrequency", "-pfreq", required=False,
                        type=int,
                        default=10,
                        help="Update loss over epoch plots after this number of epochs.")
    parser.add_argument("--multiGPUTraining", "-mgpu", required=False,
                        action="store_true",
                        help="Train on every visible GPU with MirroredStrategy. NOTE "
                             "--batchSize is the GLOBAL batch and is split across the "
                             "replicas, so pass N times the per-GPU batch you want. "
                             "Previously this was type=bool, which meant it DEMANDED a "
                             "value and then treated any non-empty string as True -- "
                             "'--multiGPUTraining False' switched multi-GPU ON. It is a "
                             "plain switch now, like every other boolean flag here.")
    parser.add_argument("--whichGPU", "-wgpu", required=False,
                        type=int,
                        default=1,
                        help="Specify which GPU to use for training in the single GPU case. "
                             "One-based: 1 is the first GPU. The default was the string \"\", "
                             "which argparse ran through type=int and rejected, so omitting this "
                             "argument aborted the run.")
    parser.add_argument("--saveMemory", "-sm", required=False,
                        action='store_true',
                        help="Save memory by not loading all data into memory at once.")
    parser.add_argument("--createDataOnly", "-cdo", required=False,
                        action='store_true',
                        default=False,
                        help="Only create TFRecords and exit (do not run training).")
    parser.add_argument("--trainOnly", "-to", required=False,
                        action='store_true',
                        default=False,
                        help="Only run training using existing TFRecords (do not create TFRecords).")
    parser.add_argument("--threads", "-t", required=False,
                        type=int,
                        default=4,
                        help="Number of threads to use for TFRecord writing.")
    parser.add_argument("--resume", "-r", required=False, action='store_true',
                        help="If set, attempt to resume training by loading the latest checkpoint from the output folder.")
    parser.add_argument("--mixedPrecision", "-mp", required=False, action='store_true',
                        help="Enable mixed-precision (float16) training. Faster and uses less GPU memory on modern GPUs; does not change system RAM usage.")
    parser.add_argument("--seed", "-sd", required=False,
                        type=int, default=None,
                        help="Seed for Python, NumPy and TensorFlow random number "
                             "generators. Set it to make a run repeatable, or to "
                             "obtain independent runs of the same configuration "
                             "by varying it. Unset leaves the generators "
                             "unseeded, which is the previous behaviour.")
    parser.add_argument("--keepTFRecords", "-k", required=False,
                        action='store_true',
                        default=False,
                        help="Do not delete TFRecord files after training.")
    parser.add_argument('--version', action='version',
                           version='%(prog)s {}'.format(__version__))

    return parser

def create_container(chrom, matrix, chromatinpath, binSize=None, inputMode="bigwig", randomSeed=42):
    container = dataContainer.DataContainer(chromosome=chrom,
                                            matrixFilePath=matrix,
                                            chromatinFolder=chromatinpath,
                                            binSize=binSize,
                                            inputMode=inputMode,
                                            randomSeed=randomSeed)
    return container

def create_data(pTrainingMatrices, 
                pTrainingChromosomes, 
                ptrainingChromatinFolders, 
                pValidationMatrices, 
                pValidationChromosomes, 
                pvalidationChromatinFolders,
                pWindowSize,
                pOutputFolder,
                pBatchSize,
                pFlipSamples,
                pFigureFileFormat,
                pRecordSize,
                noScaleNorm=False,
                pSaveMemory=True,
                pThreads=4,
                pMinTargetCoverage=0.0,
                pExcludeRegions=None,
                pScaleTargetToUnitRange=False,
                pInputMode="random2d",
                pWindowMode="diagonal",
                pGridStride=None,
                pBinSize=250000,
                pRandomSeed=42,
                pLogTransformTarget=False):
    os.makedirs(pOutputFolder, exist_ok=True)
    debugstate = None
    paramDict = locals().copy()

    trainChromNameList = [x.lstrip("chr") for x in pTrainingChromosomes]
    trainChromNameList = sorted(list(set(trainChromNameList)))
    paramDict["trainChromNameList"] = trainChromNameList

    valChromNameList = [x.lstrip("chr") for x in pValidationChromosomes]
    valChromNameList = sorted(list(set(valChromNameList)))
    paramDict["valChromNameList"] = valChromNameList

    if pInputMode == "random2d":
        if not ptrainingChromatinFolders:
            ptrainingChromatinFolders = [None] * len(pTrainingMatrices)
        if not pvalidationChromatinFolders:
            pvalidationChromatinFolders = [None] * len(pValidationMatrices)
    else:
        if not ptrainingChromatinFolders or not pvalidationChromatinFolders:
            raise SystemExit("Aborting. --trainingChromatinFolders (-tcp) and --validationChromatinFolders (-vcp) are required when --inputMode is 'bigwig'.")
        if len(pTrainingMatrices) != len(ptrainingChromatinFolders):
            msg = "Number of train matrices and chromatin paths must match\n"
            msg += "Current numbers: Matrices: {:d}; Chromatin Paths: {:d}"
            msg = msg.format(len(pTrainingMatrices), len(ptrainingChromatinFolders))
            raise SystemExit(msg)
        if len(pValidationMatrices) != len(pvalidationChromatinFolders):
            msg = "Number of validation matrices and chromatin paths must match\n"
            msg += "Current numbers: Matrices: {:d}; Chromatin Paths: {:d}"
            msg = msg.format(len(pValidationMatrices), len(pvalidationChromatinFolders))
            raise SystemExit(msg)

    log.info(
        "[CREATE-DATA] Preparing containers: inputMode=%s, windowMode=%s, gridStride=%s, windowSize=%d, "
        "trainChroms=%s, valChroms=%s",
        pInputMode, pWindowMode, str(pGridStride), pWindowSize, trainChromNameList, valChromNameList
    )
    print(
        f"[CREATE-DATA] Preparing containers: inputMode={pInputMode}, windowMode={pWindowMode}, "
        f"gridStride={pGridStride}, windowSize={pWindowSize}, "
        f"trainChroms={trainChromNameList}, valChroms={valChromNameList}",
        flush=True
    )

    traindataContainerList = []
    for chrom in trainChromNameList:
        for matrix, chromatinpath in zip(pTrainingMatrices, ptrainingChromatinFolders):
            traindataContainerList.append(
                create_container(chrom, matrix, chromatinpath, binSize=pBinSize, inputMode=pInputMode, randomSeed=pRandomSeed)
            )

    valdataContainerList = []
    for chrom in valChromNameList:
        for matrix, chromatinpath in zip(pValidationMatrices, pvalidationChromatinFolders):
            valdataContainerList.append(
                create_container(chrom, matrix, chromatinpath, binSize=pBinSize, inputMode=pInputMode, randomSeed=pRandomSeed)
            )

    #Measure the target value range ONCE, over every training matrix and
    #chromosome, and reuse it everywhere. Per-chromosome ranges would map the
    #same physical value to a different target on each chromosome.
    hasNpzSource = any(utils.isNpzMatrixSource(m) for m in pTrainingMatrices)
    useGlobalTargetRange = bool(
        pScaleTargetToUnitRange
        or pLogTransformTarget
        or (hasNpzSource and noScaleNorm)
    )
    targetValueRange = None
    if useGlobalTargetRange:
        targetValueRange = utils.observedValueRange(
            pTrainingMatrices,
            trainChromNameList,
            pLogTransform=pLogTransformTarget
        )
        rangePath = os.path.join(pOutputFolder, "target_value_range.json")
        utils.saveValueRange(rangePath, *targetValueRange, pLogTransform=pLogTransformTarget)
        msg = (
            "[TARGET-SCALE] Global target value range measured from training targets "
            "(log1p={:s}): [{:.4f}, {:.4f}] -> mapped to [0, 1] (saved to {:s})"
        ).format(str(pLogTransformTarget), targetValueRange[0], targetValueRange[1], rangePath)
        log.info(msg)
        print(msg, flush=True)
        paramDict["targetValueRange"] = targetValueRange

    flankingSize = 0 if pInputMode == "random2d" else pWindowSize
    #define the load params for the containers
    loadParams = {
        "scaleFeatures": True,
        "clampFeatures": False,
        #The two target scalings are alternatives, not a stack. The old
        #one calls utils.scaleArray on the sparse matrix, which raises
        #on any signed target; the new one supersedes it and works on
        #the dense window instead. Asking for both would crash.
        "scaleTargets": False if useGlobalTargetRange else noScaleNorm,
        "windowSize": pWindowSize,
        "flankingSize": flankingSize,
        "maximumDistance": None,
        "minTargetCoverage": pMinTargetCoverage,
        "excludedRegions": pExcludeRegions,
        "targetValueRange": targetValueRange,
        "windowMode": pWindowMode,
        "gridStride": pGridStride,
        "logTransformTarget": pLogTransformTarget,
    }
    if pExcludeRegions:
        missing = [f for f in pExcludeRegions if not os.path.isfile(f)]
        if missing:
            msg = "Aborting. BED file(s) given to --excludeRegions not found: {:s}"
            raise SystemExit(msg.format(", ".join(missing)))
    #now load the data and write TFRecords, one container at a time.
    if len(traindataContainerList) == 0:
        msg = "Exiting. No data found"
        print(msg)
        return #nothing to do
    referenceContainer = traindataContainerList[0]
    tfRecordFilenames = []
    nr_samples_list = []
    usedTrainContainerList = []
    for container in traindataContainerList + valdataContainerList:
        role = "TRAIN" if container in traindataContainerList else "VAL"
        log.info("[CREATE-DATA] Loading %s container for chr%s...", role, str(container.chromosome))
        container.loadData(**loadParams)
        if not referenceContainer.checkCompatibility(container):
            msg = "Aborting. Incompatible data"
            raise SystemExit(msg)
        nr_samples = container.getNumberSamples()
        if not nr_samples:
            #only reachable with pMinTargetCoverage > 0: the whole chromosome is a
            #gap in the target matrix. Drop it instead of writing an empty record.
            msg = "Skipping chromosome {:s}: no sample left after skipping gaps in the target"
            print(msg.format(str(container.chromosome)))
            log.warning(msg.format(str(container.chromosome)))
            continue
        writtenRecords = container.writeTFRecord(
            pOutputFolder=pOutputFolder,
            pRecordSize=pRecordSize,
            pSaveMemory=pSaveMemory,
            pThreads=pThreads
        )
        tfRecordFilenames.append(writtenRecords)
        log.info(
            "[CREATE-DATA] Wrote %d TFRecord file(s) (%d samples) for %s chr%s",
            len(writtenRecords), nr_samples, role, str(container.chromosome)
        )
        if debugstate is not None:
            if isinstance(debugstate, int):
                idx = debugstate
            else:
                idx = None
            container.plotFeatureAtIndex(idx=idx,
                                        outpath=pOutputFolder,
                                        figureFileFormat=pFigureFileFormat)
            container.saveMatrix(outputpath=pOutputFolder, index=idx)
        nr_samples_list.append(nr_samples)
        if container in traindataContainerList:
            usedTrainContainerList.append(container)
    log.info("[CREATE-DATA] All TFRecords created successfully (%d container groups).", len(tfRecordFilenames))
    print('ALL TF RECORDS CREATED!')
    if len(usedTrainContainerList) == 0:
        msg = "Exiting. No training sample left after skipping gaps in the target matrix"
        raise SystemExit(msg)
    #metadata is taken from a container that actually produced records, because
    #storedFeatures is only set by writeTFRecord
    container0 = usedTrainContainerList[0]

    #data is no longer needed
    for container in traindataContainerList + valdataContainerList:
        container.unloadData()
    
    print(tfRecordFilenames)
    print(len(tfRecordFilenames))
    print(len(usedTrainContainerList))

    #different binSizes are ok
    #not clear which binSize to use for prediction when they differ during training.
    #For now, store the max.
    binSize = max([container.binSize for container in usedTrainContainerList])

    return tfRecordFilenames, len(usedTrainContainerList), nr_samples_list, container0.storedFeatures, container0.nr_factors

def training(pTfRecordFilenames,
             pLengthTrainDataContainerList,
             pWindowSize,
             pOutputFolder,
             pEpochs,
             pLossWeightPixel,
             pLossWeightDiscriminator,
             pLossWeightAdversarial,
             pLossTypePixel,
             pLossWeightTV,
             pLearningRateGenerator,
             pLearningRateDiscriminator,
             pBeta1,
             pFigureFileFormat,
             pPlotFrequency,
             pFlipSamples,
             pScope,
             pBatchSize,
             pRecordSize,
             pStoredFeaturesDict,
             pNumberSamplesList,
             pNumberOfFactors,
             pResume=False,
             pMixedPrecision=False,
             pValidationSteps=0,
             pTargetValueRange=None,
             pInputMode="random2d",
             pWindowMode="diagonal",
             pLogTransformTarget=False
             ):

        traindataRecords = [item for sublist in pTfRecordFilenames[0:pLengthTrainDataContainerList] for item in sublist]
        valdataRecords = [item for sublist in pTfRecordFilenames[pLengthTrainDataContainerList:] for item in sublist]

        nr_trainingSamples = sum(pNumberSamplesList[0:pLengthTrainDataContainerList])
        nr_valSamples = sum(pNumberSamplesList[pLengthTrainDataContainerList:])
        log.info(
            "[TRAINING-SETUP] Training samples=%d across %d TFRecords; Validation samples=%d across %d TFRecords; "
            "inputMode=%s, windowMode=%s, windowSize=%d, batchSize=%d, epochs=%d",
            nr_trainingSamples, len(traindataRecords), nr_valSamples, len(valdataRecords),
            pInputMode, pWindowMode, pWindowSize, pBatchSize, pEpochs
        )
        print(
            f"[TRAINING-SETUP] Training samples={nr_trainingSamples} ({len(traindataRecords)} TFRecords), "
            f"Validation samples={nr_valSamples} ({len(valdataRecords)} TFRecords), "
            f"inputMode={pInputMode}, windowMode={pWindowMode}, windowSize={pWindowSize}, "
            f"batchSize={pBatchSize}, epochs={pEpochs}",
            flush=True
        )

        #build the input streams for training
        shuffleBufferSize = 3*pRecordSize
        trainDs = tf.data.TFRecordDataset(traindataRecords,
                                            num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                            compression_type=records.TFRECORD_COMPRESSION or "")
        trainDs = trainDs.map(lambda x: records.parse_function(x, pStoredFeaturesDict), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        #Map an untransformed target onto [0, 1] here rather than at write time.
        #The generator ends in a sigmoid and cannot emit a negative number, while
        #an Akita-style target is about half negative; records written with
        #--noScaleNorm hold the raw values, so one affine step at read time makes
        #them usable without rewriting hundreds of GB.
        def _toUnitRange(x, y):
            lo, hi = float(pTargetValueRange[0]), float(pTargetValueRange[1])
            raw_t = y["out_matrixData"]
            if pLogTransformTarget:
                raw_t = tf.math.log1p(tf.maximum(raw_t, 0.0))
            t = tf.clip_by_value((raw_t - lo) / (hi - lo), 0.0, 1.0)
            out = dict(y); out["out_matrixData"] = t
            return x, out

        if pTargetValueRange is not None:
            print("mapping the target onto [0, 1] at read time using "
                  "[{:.4f}, {:.4f}]".format(*pTargetValueRange))
            trainDs = trainDs.map(_toUnitRange,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if pFlipSamples:
            is_2d_inp = (pInputMode == "random2d")
            is_grid_m = (pWindowMode == "grid")
            flippedDs = trainDs.map(lambda a,b: records.mirror_function(a["factorData"], b["out_matrixData"], is_2d_input=is_2d_inp, is_grid_mode=is_grid_m))
            trainDs = trainDs.concatenate(flippedDs)
        trainDs = trainDs.shuffle(buffer_size=shuffleBufferSize, reshuffle_each_iteration=True)
        trainDs = trainDs.batch(pBatchSize, drop_remainder=True)
        trainDs = trainDs.prefetch(tf.data.experimental.AUTOTUNE)
        #build the input streams for validation
        validationDs = tf.data.TFRecordDataset(valdataRecords,
                                                num_parallel_reads=tf.data.experimental.AUTOTUNE,
                                                compression_type=records.TFRECORD_COMPRESSION or "")
        validationDs = validationDs.map(lambda x: records.parse_function(x, pStoredFeaturesDict) , num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if pTargetValueRange is not None:
            validationDs = validationDs.map(_toUnitRange,
                                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validationDs = validationDs.batch(pBatchSize)
        validationDs = validationDs.prefetch(tf.data.experimental.AUTOTUNE)
        
        steps_per_epoch = max(1, int(np.floor(nr_trainingSamples / pBatchSize)))
        if pFlipSamples:
            steps_per_epoch *= 2

        use_embedding = (pInputMode == "bigwig")
        enforce_symmetry = (pWindowMode == "diagonal")
        if use_embedding and isinstance(pStoredFeaturesDict, dict) and "factorData" in pStoredFeaturesDict:
            stored_shape = pStoredFeaturesDict["factorData"].get("shape")
            if stored_shape is not None and len(stored_shape) >= 2:
                pNumberOfFactors = int(stored_shape[1])

        hicGanModel = hicGAN.HiCGAN(log_dir=pOutputFolder, 
                                        number_factors=pNumberOfFactors,
                                        loss_weight_pixel=pLossWeightPixel,
                                        loss_weight_adversarial=pLossWeightAdversarial,
                                        loss_weight_discriminator=pLossWeightDiscriminator, 
                                        loss_type_pixel=pLossTypePixel, 
                                        loss_weight_tv=pLossWeightTV, 
                                        input_size=pWindowSize,
                                        learning_rate_generator=pLearningRateGenerator,
                                        learning_rate_discriminator=pLearningRateDiscriminator,
                                        adam_beta_1=pBeta1,
                                        plot_type=pFigureFileFormat,
                                        plot_frequency=pPlotFrequency,
                                        scope=pScope,
                                        mixed_precision=pMixedPrecision,
                                        restore_checkpoint=pResume,
                                        use_embedding=use_embedding,
                                        enforce_symmetry=enforce_symmetry)
        
        hicGanModel.plotModels(pOutputPath=pOutputFolder, pFigureFileFormat=pFigureFileFormat)

        log.info("Starting training at %s" % datetime.now())
        hicGanModel.fit(train_ds=trainDs, epochs=pEpochs, test_ds=validationDs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=pValidationSteps)
        log.info("Training finished at %s" % datetime.now())


def delete_model_files(pTFRecordFiles):
    log.info("Cleaning up temporary files...")
    print(pTFRecordFiles)
    for tfRecordfile in pTFRecordFiles:
        for file in tfRecordfile:
            if os.path.exists(file):
                os.remove(file)

def main(args=None):
    args = parse_arguments().parse_args(args)
    os.makedirs(args.outputFolder, exist_ok=True)
    utils.configureLogging(os.path.join(args.outputFolder, "training.log"))

    inputMode = args.inputMode
    if inputMode is None:
        inputMode = "random2d" if not args.trainingChromatinFolders else "bigwig"
    windowMode = args.windowMode
    randomSeed = args.seed if args.seed is not None else 42

    log.info(
        "[MAIN] Starting hicTraining: inputMode=%s, windowMode=%s, gridStride=%s, windowSize=%d, "
        "binSize=%d, logTransformTarget=%s, seed=%s",
        inputMode, windowMode, str(args.gridStride), args.windowSize,
        args.binSize, str(args.logTransformTarget), str(args.seed)
    )

    #seed every generator that can affect a run: weight initialisation, the
    #shuffle buffer and dropout all draw from these
    if args.seed is not None:
        import random
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        print("random seed set to %d" % args.seed)

    # Resolve --targetValueRange once, here, so the training path gets a plain
    # (min, max) tuple or None.
    resolvedTargetRange = None
    if args.targetValueRange:
        if args.scaleTargetToUnitRange:
            print("Exiting. --targetValueRange maps the target at READ time and "
                  "--scaleTargetToUnitRange at WRITE time; using both would apply "
                  "the mapping twice. Choose one.")
            return
        if args.targetValueRange.lower() == "auto":
            resolvedTargetRange = utils.observedValueRange(
                args.trainingMatrices,
                [c.lstrip("chr") for c in args.trainingChromosomes],
                pLogTransform=args.logTransformTarget)
            os.makedirs(args.outputFolder, exist_ok=True)
            utils.saveValueRange(
                os.path.join(args.outputFolder, "target_value_range.json"),
                *resolvedTargetRange,
                pLogTransform=args.logTransformTarget)
        elif os.path.exists(args.targetValueRange):
            resolvedTargetRange = utils.loadValueRange(args.targetValueRange)
        else:
            try:
                lo, hi = (float(x) for x in args.targetValueRange.split(","))
                resolvedTargetRange = (lo, hi)
            except Exception:
                print("Exiting. --targetValueRange must be 'auto', a path to "
                      "target_value_range.json, or 'min,max'; got "
                      f"{args.targetValueRange!r}")
                return
        print("target value range for the read-time mapping: "
              "[{:.4f}, {:.4f}]".format(*resolvedTargetRange))

    # Reject mutually exclusive flags: both cannot be true at the same time
    if args.createDataOnly and args.trainOnly:
        print("Exiting. Flags --createDataOnly and --trainOnly are mutually exclusive; choose only one or neither.")
        return

    for matrix in args.trainingMatrices + args.validationMatrices:
        if not os.path.exists(matrix):
            msg = "Exiting. Matrix file or folder not found: {:s}".format(matrix)
            print(msg)
            log.error(msg)
            return
        if not (matrix.endswith(".cool") or utils.isNpzMatrixSource(matrix)):
            msg = "Exiting. Only .cool files, .npz files, or directories of .npz files are supported: {:s}".format(matrix)
            print(msg)
            log.error(msg)
            return

    if (args.trainOnly and not args.createDataOnly) or (not args.createDataOnly and not args.trainOnly):
        physical_gpus = tf.config.list_physical_devices('GPU')

        try:
            if not physical_gpus:
                log.info("No physical GPUs detected. Falling back to CPU strategy (/CPU:0).")
                print("[DEVICE] No physical GPUs detected. Running on /CPU:0.", flush=True)
                strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
            elif args.multiGPUTraining:
                # --- MULTI GPU SETUP ---
                tf.config.set_visible_devices(physical_gpus, 'GPU')
                for gpu_device in physical_gpus:
                    tf.config.experimental.set_memory_growth(gpu_device, True)
                strategy = tf.distribute.MirroredStrategy()
                log.info(f"Using MirroredStrategy on {len(physical_gpus)} GPUs")
            else:
                # --- SINGLE GPU SETUP ---
                log.info("Using single GPU training")
                target_index = args.whichGPU - 1
                if target_index < 0 or target_index >= len(physical_gpus):
                    raise ValueError(f"Invalid GPU index: {target_index}. Available: {len(physical_gpus)}")

                tf.config.set_visible_devices(physical_gpus[target_index], 'GPU')
                tf.config.experimental.set_memory_growth(physical_gpus[target_index], True)
                log.info("Physically selected GPU: {}".format(physical_gpus[target_index].name))
                strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")

        except Exception as e:
            print("Error during GPU setup: {}".format(e))
            raise e

        if args.mixedPrecision and physical_gpus:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            log.info("Mixed precision enabled: global policy = %s", tf.keras.mixed_precision.global_policy().name)
    # Only force the CPU strategy for a *pure* create-data run. In "all" mode
    # the GPU strategy selected above must survive, otherwise this line would
    # overwrite it and training would silently fall back to the CPU (~300 s/iter).
    if args.createDataOnly and not args.trainOnly:
        strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")


    with strategy.scope() as scope: 

        if (not args.trainOnly and args.createDataOnly) or (not args.createDataOnly and not args.trainOnly):
            tfRecordFilenames, traindataContainerListLength, nr_samples_list, storedFeatures, nr_factors = create_data(
                args.trainingMatrices, 
                args.trainingChromosomes, 
                args.trainingChromatinFolders, 
                args.validationMatrices, 
                args.validationChromosomes, 
                args.validationChromatinFolders,
                args.windowSize,
                args.outputFolder,
                args.batchSize,
                args.flipSamples,
                args.figureFileFormat,
                args.recordSize,
                args.noScaleNorm,
                args.saveMemory,
                args.threads,
                pMinTargetCoverage=args.minTargetCoverage,
                pExcludeRegions=args.excludeRegions,
                pScaleTargetToUnitRange=args.scaleTargetToUnitRange,
                pInputMode=inputMode,
                pWindowMode=windowMode,
                pGridStride=args.gridStride,
                pBinSize=args.binSize,
                pRandomSeed=randomSeed,
                pLogTransformTarget=args.logTransformTarget
            )
            meta = {
                "tfRecordFilenames": tfRecordFilenames,
                "traindataContainerListLength": traindataContainerListLength,
                "nr_samples_list": nr_samples_list,
                "storedFeatures": storedFeatures,
                "nr_factors": nr_factors,
                "inputMode": inputMode,
                "windowMode": windowMode,
                "gridStride": args.gridStride,
                "binSize": args.binSize,
                "windowSize": args.windowSize,
                "logTransformTarget": args.logTransformTarget,
                "seed": randomSeed,
                "created_at": datetime.utcnow().isoformat()
            }
            outfile = os.path.join(args.outputFolder, "hicgan_metadata.pkl")
            with open(outfile, "wb") as f:
                pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.info("[MAIN] Saved metadata to %s", outfile)
            print(f"Saved metadata to {outfile}")
        if (args.trainOnly and not args.createDataOnly) or (not args.createDataOnly and not args.trainOnly):
            if args.trainOnly:
                meta_file = os.path.join(args.outputFolder, "hicgan_metadata.pkl")
                if not os.path.exists(meta_file):
                    print(f"Metadata file not found: {meta_file}. Cannot proceed with training.")
                    return
                with open(meta_file, "rb") as f:
                    meta = pickle.load(f)
                tfRecordFilenames = meta["tfRecordFilenames"]
                traindataContainerListLength = meta["traindataContainerListLength"]
                nr_samples_list = meta["nr_samples_list"]
                storedFeatures = meta["storedFeatures"]
                nr_factors = meta["nr_factors"]
                inputMode = meta.get("inputMode", inputMode)
                windowMode = meta.get("windowMode", windowMode)
            training(
                pTfRecordFilenames=tfRecordFilenames,
                pLengthTrainDataContainerList=traindataContainerListLength,
                pWindowSize=args.windowSize,
                pOutputFolder=args.outputFolder,
                pEpochs=args.epochs,
                pLossWeightPixel=args.lossWeightPixel,
                pLossWeightDiscriminator=args.lossWeightDiscriminator,
                pLossWeightAdversarial=args.lossWeightAdversarial,
                pLossTypePixel=args.lossTypePixel,
                pLossWeightTV=args.lossWeightTV,
                pLearningRateGenerator=args.learningRateGenerator,
                pLearningRateDiscriminator=args.learningRateDiscriminator,
                pBeta1=args.beta1,
                pFigureFileFormat=args.figureFileFormat,
                pPlotFrequency=args.plotFrequency,
                pFlipSamples=args.flipSamples,
                pScope=scope,
                pBatchSize=args.batchSize,
                pRecordSize=args.recordSize,
                pStoredFeaturesDict=storedFeatures,
                pNumberSamplesList=nr_samples_list,
                pNumberOfFactors=nr_factors,
                pResume=args.resume,
                pMixedPrecision=args.mixedPrecision,
                pValidationSteps=args.validationSteps,
                pTargetValueRange=resolvedTargetRange,
                pInputMode=inputMode,
                pWindowMode=windowMode,
                pLogTransformTarget=args.logTransformTarget
            )
        if not args.keepTFRecords and ((args.trainOnly and not args.createDataOnly) or (not args.createDataOnly and not args.trainOnly)):
            delete_model_files(pTFRecordFiles=tfRecordFilenames)