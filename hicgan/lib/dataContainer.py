import os
import numpy as np
from tensorflow import dtypes as tfdtypes
from scipy.sparse import save_npz, csr_matrix
from tqdm import tqdm
import concurrent.futures

from . import utils
from . import records


import logging
log = logging.getLogger(__name__)


class DataContainer():
    def __init__(self, chromosome, matrixFilePath, chromatinFolder, binSize=None,
                 inputMode=None, numBins=None, randomSeed=None):
        self.chromosome = str(chromosome)
        self.matrixFilePath = matrixFilePath
        self.chromatinFolder = chromatinFolder
        if inputMode is None:
            if chromatinFolder is None or str(chromatinFolder).lower() in ("random", "random2d", "none", ""):
                self.inputMode = "random2d"
            else:
                self.inputMode = "bigwig"
        else:
            self.inputMode = str(inputMode).lower()
        self.FactorDataArray = None
        self.RandomMatrix2D = None
        self.nr_factors = None
        self.sparseHiCMatrix = None
        self.sequenceArray = None
        self.binSize = binSize
        self.numBins = int(numBins) if numBins is not None else None
        self.original_num_bins = int(numBins) if numBins is not None else None
        self.padded_bins = 0
        self.randomSeed = randomSeed
        self.factorNames = None
        self.prefixDict_factors = None
        self.prefixDict_matrix = None
        self.prefixDict_sequence = None
        self.chromSize_factors = None
        self.chromSize_matrix = None
        self.storedFeatures = None
        self.storedFiles = None
        self.windowSize = None
        self.flankingSize = None
        self.maximumDistance = None
        self.windowMode = "diagonal"
        self.gridStride = None
        self.gridCoordinates = None
        self.padShortChromosomes = True
        self.logTransformTarget = False
        self.minTargetCoverage = 0.0
        self.excludedRegions = None
        self.includedRegions = None
        self.targetValueRange = None
        self.sampleIndices = None
        self.data_loaded = False

    def __loadFactorData(self, ignoreChromLengths=False, scaleFeatures=False, clampFeatures=False):
        if self.inputMode == "random2d":
            nr_bins = None
            if self.sparseHiCMatrix is not None:
                nr_bins = int(self.sparseHiCMatrix.shape[0])
            elif self.numBins is not None:
                nr_bins = int(self.numBins)
            elif self.chromatinFolder is not None and utils.isNpzMatrixSource(self.chromatinFolder):
                chromBins = utils.getChromBinsFromNpzSource(self.chromatinFolder)
                nr_bins = chromBins.get(str(self.chromosome).lstrip("chr"))
            if nr_bins is None or nr_bins <= 0:
                msg = (
                    f"Cannot determine bin count (n_bins) for chromosome '{self.chromosome}' in random2d mode. "
                    f"Provide a target .npz/.cool matrix, numBins, or --chromSizesPath pointing to chrom_index.csv / .npz folder."
                )
                raise ValueError(msg)
            if self.original_num_bins is None:
                self.original_num_bins = nr_bins
            self.numBins = nr_bins
            if self.binSize is None:
                self.binSize = utils.inferBinSizeFromPath(self.matrixFilePath or self.chromatinFolder, 250000)
            self.chromSize_factors = int(nr_bins * self.binSize)
            self.RandomMatrix2D = utils.generateSymmetricRandomMatrix(
                pNumBins=nr_bins, pSeed=self.randomSeed, pChrom=self.chromosome
            )
            self.FactorDataArray = self.RandomMatrix2D
            self.factorNames = ["random2d"]
            self.nr_factors = 1
            return

        #load chromatin factor data from bigwig files
        if self.chromatinFolder is None:
            return
        #ensure that binSizes for matrix (if given) and factors match
        if self.binSize is None:
            msg = "No binSize given; use a Hi-C matrix or explicitly specify binSize for the container"   
            raise TypeError(msg)
        ###load data for a specific chromsome
        #get the names of the bigwigfiles
        bigwigFileList = utils.getBigwigFileList(self.chromatinFolder)
        bigwigFileList = sorted(bigwigFileList)
        if len(bigwigFileList) is None or len(bigwigFileList) == 0:
            msg = "Warning: folder {:s} does not contain any bigwig files"
            msg = msg.format(self.chromatinFolder)
            print(msg)
            return
        #check the chromosome name prefixes (e.g. "" or "chr") and sizes
        chromSizeList = []
        prefixDict_factors = dict()
        for bigwigFile in bigwigFileList:
            try:
                prefixDict_factors[bigwigFile] = utils.getChromPrefixBigwig(bigwigFile)
                chromname = prefixDict_factors[bigwigFile] + self.chromosome
                chromSizeList.append( utils.getChromSizesFromBigwig(bigwigFile)[chromname] )
            except Exception as e:
                msg = str(e) + "\n"
                msg += "Could not load data from bigwigfile {}".format(bigwigFile) 
                raise IOError(msg)
        #the chromosome lengths should be equal in all bigwig files
        if len(set(chromSizeList)) != 1 and not ignoreChromLengths:
            msg = "Invalid data. Chromosome lengths differ in bigwig files:"
            for i, filename in enumerate(bigwigFileList):
                msg += "{:s}: {:d}\n".format(filename, chromSizeList[i])
            raise IOError(msg)
        elif len(set(chromSizeList)) != 1 and ignoreChromLengths:
            chromSize_factors = min(chromSizeList)
        else:
            chromSize_factors = chromSizeList[0]
        nr_bins = int( np.ceil(chromSize_factors / self.binSize) )
        if self.sparseHiCMatrix is not None and not utils.isNpzMatrixSource(self.matrixFilePath):
            if self.chromSize_matrix is not None and self.chromSize_matrix != chromSize_factors:
                msg = "Chrom lengths not equal between matrix and bigwig files\n"
                msg += "Matrix: {:d} -- Factors: {:d}".format(self.chromSize_matrix, chromSize_factors)
                raise IOError(msg)
        #load the data into memory now
        self.factorNames = [os.path.splitext(os.path.basename(name))[0] for name in bigwigFileList]
        base_factors = len(self.factorNames)
        self.nr_factors = base_factors * 2 if self.windowMode == "grid" else base_factors
        self.prefixDict_factors = prefixDict_factors
        self.chromSize_factors = chromSize_factors
        self.FactorDataArray = np.empty(shape=(len(bigwigFileList),nr_bins))
        if self.windowMode == "grid":
            msg = "Loaded {:d} chromatin features (x2 for grid row/col pairs = {:d} input channels) from folder {:s}\n"
            msg = msg.format(base_factors, self.nr_factors, self.chromatinFolder)
        else:
            msg = "Loaded {:d} chromatin features from folder {:s}\n"
            msg = msg.format(self.nr_factors, self.chromatinFolder)
        featLoadedMsgList = [] #pretty printing for features loaded

        def process_bigwig_file(bigwigFile):
            chromname = self.prefixDict_factors[bigwigFile] + self.chromosome
            tmpArray = utils.binChromatinFactor(pBigwigFileName=bigwigFile,
                                                pBinSizeInt=self.binSize,
                                                pChromStr=chromname,
                                                pChromSize=self.chromSize_factors)
            if clampFeatures:
                tmpArray = utils.clampArray(tmpArray)
            if scaleFeatures:
                tmpArray = utils.scaleArray(tmpArray)
            return tmpArray

        results = [None] * len(bigwigFileList)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futureToIndex = {executor.submit(process_bigwig_file, bigwigFile): i
                             for i, bigwigFile in enumerate(bigwigFileList)}
            for future in concurrent.futures.as_completed(futureToIndex):
                results[futureToIndex[future]] = future.result()
        for i, tmpArray in enumerate(results):
            self.FactorDataArray[i] = tmpArray
            nr_nonzero_abs = np.count_nonzero(tmpArray)
            nr_nonzero_perc = nr_nonzero_abs / tmpArray.size * 100
            msg2 = "{:s} - min. {:.3f} - max. {:.3f} - nnz. {:d} ({:.2f}%)"
            msg2 = msg2.format(bigwigFileList[i], tmpArray.min(), tmpArray.max(), nr_nonzero_abs, nr_nonzero_perc)
            featLoadedMsgList.append(msg2)
        self.FactorDataArray = np.transpose(self.FactorDataArray)
        print(msg + "\n".join(featLoadedMsgList))

        # Align BigWig bin count with .npz target matrix (during training) or numBins from chrom_index.csv (during prediction)
        target_bins = None
        if self.sparseHiCMatrix is not None and utils.isNpzMatrixSource(self.matrixFilePath):
            target_bins = int(self.sparseHiCMatrix.shape[0])
        elif self.numBins is not None:
            target_bins = int(self.numBins)

        if target_bins is not None and self.FactorDataArray.shape[0] != target_bins:
            bw_bins = int(self.FactorDataArray.shape[0])
            if bw_bins > target_bins:
                self.FactorDataArray = self.FactorDataArray[:target_bins, :]
                align_msg = (
                    f"[BIGWIG-ALIGN] Chromosome {self.chromosome}: BigWig bins ({bw_bins}) > target .npz bins ({target_bins}). "
                    f"Cropped BigWig features to {target_bins} bins."
                )
            else:
                pad_bw = target_bins - bw_bins
                self.FactorDataArray = np.pad(
                    self.FactorDataArray,
                    ((0, pad_bw), (0, 0)),
                    mode="constant",
                    constant_values=0.0
                )
                align_msg = (
                    f"[BIGWIG-ALIGN] Chromosome {self.chromosome}: BigWig bins ({bw_bins}) < target .npz bins ({target_bins}). "
                    f"Zero-padded BigWig features by +{pad_bw} bins to {target_bins} bins."
                )
            log.info(align_msg)
            print(align_msg, flush=True)
            nr_bins = target_bins
            self.chromSize_factors = int(nr_bins * self.binSize)

        if self.original_num_bins is None:
            self.original_num_bins = nr_bins
        self.numBins = nr_bins
            
    def __loadMatrixData(self, scaleMatrix=False):
        #load target matrix from .npz folder/file or cooler file
        if self.matrixFilePath is None:
            return
        is_npz = utils.isNpzMatrixSource(self.matrixFilePath)
        try:
            if is_npz:
                sparseHiCMatrix, binSize = utils.getMatrixFromNpz(
                    self.matrixFilePath, self.chromosome, pBinSizeInt=self.binSize
                )
                chromsize_matrix = int(sparseHiCMatrix.shape[0] * binSize)
            else:
                prefixDict_matrix = {self.matrixFilePath: utils.getChromPrefixCooler(self.matrixFilePath)}
                chromname = prefixDict_matrix[self.matrixFilePath] + self.chromosome
                chromsize_matrix = utils.getChromSizesFromCooler(self.matrixFilePath)[chromname]
                sparseHiCMatrix, binSize = utils.getMatrixFromCooler(self.matrixFilePath, chromname)
        except Exception as e:
            msg = "Error: Could not load data from target matrix {:s} for chr{:s}: {:s}"
            msg = msg.format(str(self.matrixFilePath), str(self.chromosome), str(e))
            raise IOError(msg)
        #scale to 0..1, if requested
        if scaleMatrix:
            sparseHiCMatrix = utils.scaleArray(sparseHiCMatrix)       
        #ensure that chrom sizes for matrix and factors are the same
        if self.chromSize_factors is not None and not is_npz and self.chromSize_matrix != chromsize_matrix:
            if self.chromSize_factors != chromsize_matrix:
                msg = "Chromsize of matrix does not match bigwig files\n"
                msg += "Matrix: {:d} -- Bigwig files: {:d}"
                msg = msg.format(chromsize_matrix, self.chromSize_factors)
                raise IOError(msg)
        self.chromSize_matrix = chromsize_matrix
        self.original_num_bins = int(sparseHiCMatrix.shape[0])
        self.numBins = int(sparseHiCMatrix.shape[0])
        #ensure that binSizes for matrix and factors (if given) match
        if self.binSize is None or self.binSize == binSize:
            self.binSize = binSize
            self.sparseHiCMatrix = sparseHiCMatrix
        elif self.binSize is not None and self.binSize != binSize:
            msg = "Matrix has wrong binSize\n"
            msg += "Matrix: {:d} -- Binned chromatin factors {:d}"
            msg = msg.format(binSize, self.binSize)
            raise IOError(msg)
        src_type = "npz" if is_npz else "cooler"
        dmin = float(self.sparseHiCMatrix.min()) if self.sparseHiCMatrix.nnz > 0 else 0.0
        dmax = float(self.sparseHiCMatrix.max()) if self.sparseHiCMatrix.nnz > 0 else 0.0
        msg = "Loaded {:s} matrix {:s}\n".format(src_type, str(self.matrixFilePath))
        msg += "chr. {:s}, matshape {:d}*{:d} -- min. {:.3f} -- max. {:.3f} -- nnz. {:d}"
        msg = msg.format(self.chromosome, self.sparseHiCMatrix.shape[0], self.sparseHiCMatrix.shape[1], dmin, dmax, self.sparseHiCMatrix.getnnz() )
        log.info(msg.replace("\n", " | "))
        print(msg)
    
    def __unloadFactorData(self):
        #unload chromatin factor data to save memory, but do not touch metadata 
        self.FactorDataArray = None
        self.RandomMatrix2D = None
        
    def __unloadMatrixData(self):
        #unload matrix data to save memory, but do not touch metadata
        self.sparseHiCMatrix = None

    def unloadData(self):
        #unload all data to save memory, but do not touch metadata
        self.__unloadFactorData()
        self.__unloadMatrixData()
        self.windowSize = None
        self.flankingSize = None
        self.maximumDistance = None
        self.sampleIndices = None
        self.gridCoordinates = None
        self.excludedRegions = None
        self.includedRegions = None
        self.targetValueRange = None
        self.data_loaded = False

    def __applyPaddingIfNeeded(self):
        """Automatically pad short chromosomes so they have at least windowSize + 2*flankingSize bins, and log details."""
        required_bins = int(self.windowSize + 2 * self.flankingSize)
        n_bins = self.__getNumberBins()
        if self.original_num_bins is None and n_bins > 0:
            self.original_num_bins = int(n_bins)

        if n_bins <= 0 or n_bins >= required_bins or not self.padShortChromosomes:
            self.padded_bins = 0
            log.info(
                "[PADDING] Chromosome %s: original_bins=%d >= required_bins=%d (windowSize=%d, flankingSize=%d) -> no padding needed.",
                str(self.chromosome), n_bins, required_bins, self.windowSize, self.flankingSize
            )
            return

        pad_bins = required_bins - n_bins
        self.padded_bins = int(pad_bins)

        if self.sparseHiCMatrix is not None:
            csr = self.sparseHiCMatrix.tocsr()
            padded_indptr = np.concatenate([
                csr.indptr,
                np.full(pad_bins, csr.indptr[-1], dtype=csr.indptr.dtype)
            ])
            self.sparseHiCMatrix = csr_matrix(
                (csr.data.copy(), csr.indices.copy(), padded_indptr),
                shape=(required_bins, required_bins),
                dtype=csr.dtype
            )

        if self.inputMode == "random2d" and self.RandomMatrix2D is not None:
            self.RandomMatrix2D = np.pad(
                self.RandomMatrix2D,
                ((0, pad_bins), (0, pad_bins)),
                mode="constant",
                constant_values=0.0
            )
            self.FactorDataArray = self.RandomMatrix2D
        elif self.FactorDataArray is not None:
            self.FactorDataArray = np.pad(
                self.FactorDataArray,
                ((0, pad_bins), (0, 0)),
                mode="constant",
                constant_values=0.0
            )

        self.numBins = required_bins
        if self.binSize is not None:
            self.chromSize_factors = int(required_bins * self.binSize)
            if self.chromSize_matrix is not None:
                self.chromSize_matrix = int(required_bins * self.binSize)

        msg = (
            f"[PADDING] Chromosome {self.chromosome}: original_bins={n_bins} < required_bins={required_bins} "
            f"(windowSize={self.windowSize}, flankingSize={self.flankingSize}, inputMode={self.inputMode}, "
            f"windowMode={self.windowMode}). Added +{pad_bins} zero-padded bins at chromosome end -> "
            f"new_shape=({required_bins}, {required_bins})."
        )
        log.info(msg)
        print(msg, flush=True)

    def __buildGridCoordinates(self):
        """Build 2D upper-triangle grid coordinates (row_start, col_start) when windowMode == 'grid'."""
        if self.windowMode != "grid":
            self.gridCoordinates = None
            return
        n_bins = self.__getNumberBins()
        cutout = int(self.windowSize + 2 * self.flankingSize)
        max_start = max(0, n_bins - cutout)
        stride = int(self.gridStride) if (self.gridStride is not None and int(self.gridStride) > 0) else max(1, self.windowSize // 2)
        self.gridStride = stride
        starts = list(range(0, max_start + 1, stride))
        if len(starts) == 0 or starts[-1] != max_start:
            starts.append(max_start)
        coords = []
        for i_idx, r in enumerate(starts):
            for c in starts[i_idx:]:
                coords.append((r, c))
        self.gridCoordinates = np.asarray(coords, dtype=np.int64)
        msg = (
            f"[WINDOW-GRID] Chromosome {self.chromosome}: windowMode=grid, windowSize={self.windowSize}, "
            f"gridStride={stride}, 1D_positions={len(starts)}, upper_triangle_windows={len(coords)} "
            f"(matrix_bins={n_bins}, original_bins={self.original_num_bins})"
        )
        log.info(msg)
        print(msg, flush=True)

    def loadData(self, windowSize, flankingSize=None, maximumDistance=None, scaleFeatures=False, clampFeatures=False,
                 scaleTargets=False, minTargetCoverage=0.0, excludedRegions=None, includedRegions=None,
                 targetValueRange=None, inputMode=None, windowMode="diagonal", gridStride=None,
                 padShortChromosomes=True, logTransformTarget=False, randomSeed=None):
        if not isinstance(windowSize, int):
            msg = "windowSize must be integer"
            raise TypeError(msg)
        if inputMode is not None:
            self.inputMode = str(inputMode).lower()
        if randomSeed is not None:
            self.randomSeed = randomSeed
        self.windowMode = str(windowMode).lower() if windowMode else "diagonal"
        self.gridStride = gridStride
        self.padShortChromosomes = bool(padShortChromosomes)
        self.logTransformTarget = bool(logTransformTarget)
        if flankingSize is None:
            flankingSize = 0 if self.inputMode == "random2d" else windowSize
        if isinstance(maximumDistance, int):
            maximumDistance = np.clip(maximumDistance, a_min=1, a_max=windowSize)
        self.windowSize = windowSize
        self.flankingSize = int(flankingSize)
        self.maximumDistance = maximumDistance
        self.minTargetCoverage = float(minTargetCoverage)
        self.excludedRegions = excludedRegions
        self.includedRegions = includedRegions
        self.targetValueRange = targetValueRange

        self.__loadMatrixData(scaleMatrix=scaleTargets)
        self.__loadFactorData(scaleFeatures=scaleFeatures, clampFeatures=clampFeatures)
        self.__applyPaddingIfNeeded()
        self.__buildGridCoordinates()
        self.data_loaded = True
        self.__computeSampleIndices()

    def checkCompatibility(self, containerIterable):
        ret = []
        try:
           for container in containerIterable:
               ret.append(self.__checkCompatibility(container))
        except:
            ret = [self.__checkCompatibility(containerIterable)]
        return np.all(ret)
        
    def __checkCompatibility(self, container):
        if not isinstance(container, DataContainer):
            return False
        if not self.data_loaded or not container.data_loaded:
            return False
        #check if the same kind of data is available for all containers
        factorsOK = type(self.FactorDataArray) == type(container.FactorDataArray)
        matrixOK = type(self.sparseHiCMatrix) == type(container.sparseHiCMatrix)
        #check if windowSize, flankingSize and maximumDistance match
        windowSizeOK = self.windowSize == container.windowSize
        flankingSizeOK = self.flankingSize == container.flankingSize
        maximumDistanceOK = self.maximumDistance == container.maximumDistance
        modeOK = (self.inputMode == container.inputMode) and (self.windowMode == container.windowMode)
        log.debug("Factors: {:s} -- Matrix: {:s} -- windowSize: {:s} -- flankingSize: {:s} -- maximumDistance: {:s}".format(str(factorsOK), str(matrixOK), str(windowSizeOK), str(flankingSizeOK), str(maximumDistanceOK)))
        log.debug("Chromatin folder: {:s} -- Nr. factors: {:s}".format(str(self.chromatinFolder), str(self.nr_factors)))
        log.debug("Chromatin folder: {:s} -- Nr. factors: {:s}".format(str(container.chromatinFolder), str(container.nr_factors)))
        #sanity check loading of bigwig files
        if self.inputMode == "bigwig" and self.chromatinFolder is not None and self.nr_factors is None:
            return False
        if container.inputMode == "bigwig" and container.chromatinFolder is not None and container.nr_factors is None:
            return False
        #if chromatin factors are present, the numbers and names of chromatin factors must match
        factorsOK = factorsOK and (self.nr_factors == container.nr_factors)
        # factorsOK = factorsOK and (self.factorNames == container.factorNames)
        log.debug("self.nr_factors: {:d} -- container.nr_factors: {:d}".format(self.nr_factors, container.nr_factors))
        log.debug("self.factorNames: {:s} -- container.factorNames: {:s}".format(str(self.factorNames), str(container.factorNames)))
        return factorsOK and matrixOK and windowSizeOK and flankingSizeOK and maximumDistanceOK and modeOK
        
    def writeTFRecord(self, pOutputFolder, pRecordSize=None, pSaveMemory=False, pThreads=4):
        '''
        Write a dataset to disk in tensorflow TFRecord format
        
        Parameters:
            pwindowSize (int): size of submatrices
            pOutfolder (str): directory where TFRecords will be written
            pflankingSize (int): size of flanking regions left/right of submatrices
            pmaximumDistance (int): cut the matrices off at this distance (in bins)
            pRecordsize (int): split the TFRecords into multiple files containing approximately this number of samples
        
        Returns:
            list of filenames written
        '''

        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to write"
            print(msg)
            return None
        nr_samples = self.getNumberSamples()
        if not nr_samples:
            msg = "Warning: chromosome {:s} has no sample, no TFRecord written"
            print(msg.format(str(self.chromosome)))
            self.storedFiles = []
            return []
        #adjust record size (yields smaller files and reduces memory load)
        recordsize = nr_samples
        if pRecordSize is not None and pRecordSize < recordsize:
            recordsize = pRecordSize
        #compute number of record files, number of samples 
        #in each file and corresponding indices
        nr_files = int( np.ceil(nr_samples/recordsize) )
        target_ct = int( np.floor(nr_samples/nr_files) )
        samples_per_file = [target_ct]*(nr_files-1) + [nr_samples-(nr_files-1)*target_ct]
        sample_indices = [sum(samples_per_file[0:i]) for i in range(len(samples_per_file)+1)] 
        #write the single files
        rawFolder = self.chromatinFolder if self.chromatinFolder else f"{self.inputMode}_{self.windowMode}"
        folderName = str(rawFolder).strip("/").replace("/","_")
        recordfiles = [os.path.join(pOutputFolder, "{:s}_{:s}_{:03d}.tfrecord".format(folderName, str(self.chromosome), i + 1)) for i in range(nr_files)]
        log.info(
            "[TFRECORD] Writing %d samples for chr%s across %d file(s) (inputMode=%s, windowMode=%s)",
            nr_samples, str(self.chromosome), nr_files, self.inputMode, self.windowMode
        )

        def storeTFRecord(recordfile, firstIndex, lastIndex, outfolder):
            log.debug("Prepare dict...")
            recordDict, storedFeaturesDict = self.__prepareWriteoutDict(pFirstIndex=firstIndex, 
                                                                        pLastIndex=lastIndex, 
                                                                        pOutfolder=outfolder)
            log.debug("Prepare dict... DONE!")
            log.debug("Write TFRecord...")
            records.writeTFRecord(pFilename=recordfile, pRecordDict=recordDict)
            log.debug("Write TFRecord... DONE!")

            return storedFeaturesDict

        storedFeaturesDictList = []
        if pSaveMemory or pThreads == 1:
            for i, (recordfile, firstIndex, lastIndex) in enumerate(zip(recordfiles, sample_indices, sample_indices[1:])):
                log.debug("Processing record file {:d} / {:d}...".format(i+1, len(recordfiles)))
                storedFeaturesDict = storeTFRecord(recordfile, firstIndex, lastIndex, pOutputFolder)
                storedFeaturesDictList.append(storedFeaturesDict)
        else:        
            max_workers = pThreads
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = [executor.submit(storeTFRecord, recordfile, firstIndex, lastIndex, pOutputFolder) for recordfile, firstIndex, lastIndex in zip(recordfiles, sample_indices, sample_indices[1:])]
                for future in concurrent.futures.as_completed(results):
                    storedFeaturesDict = future.result()
                    storedFeaturesDictList.append(storedFeaturesDict)
        self.storedFiles = recordfiles
        self.storedFeatures = storedFeaturesDict
        return recordfiles

    def hasEnoughBins(self):
        '''
        True if this chromosome has enough bins (at the current binSize) to
        produce at least one sliding-window sample for the configured
        windowSize/flankingSize.
        '''
        if not self.data_loaded or self.FactorDataArray is None:
            return False
        required_bins = self.windowSize + 2 * self.flankingSize
        return self.FactorDataArray.shape[0] >= required_bins

    def getBinCoverage(self):
        '''
        Boolean array, one entry per bin of the Hi-C matrix, True where the bin
        has at least one contact.
        '''
        if self.sparseHiCMatrix is None:
            return None
        rowNnz = np.asarray((self.sparseHiCMatrix != 0).sum(axis=1)).ravel()
        return rowNnz > 0

    def __binsFromBed(self, pPathList, pLabel):
        if not pPathList:
            return None
        pathList = pPathList
        if isinstance(pathList, str):
            pathList = [pathList]
        nr_bins = self.__getNumberBins()
        if not nr_bins:
            return None
        marked = np.zeros(nr_bins, dtype=bool)
        bareName = str(self.chromosome)
        acceptedNames = {bareName, "chr" + bareName.lstrip("chr")}
        nr_regions = 0
        for path in pathList:
            with open(path, "r") as bedfile:
                for lineNr, line in enumerate(bedfile, start=1):
                    line = line.strip()
                    if not line or line.startswith(("#", "track", "browser")):
                        continue
                    fields = line.split()
                    if len(fields) < 3:
                        msg = "Skipping line {:d} of {:s}: fewer than 3 columns"
                        log.warning(msg.format(lineNr, path))
                        continue
                    if fields[0] not in acceptedNames:
                        continue
                    try:
                        start, end = int(fields[1]), int(fields[2])
                    except ValueError:
                        msg = "Skipping line {:d} of {:s}: start/end not integers"
                        log.warning(msg.format(lineNr, path))
                        continue
                    if end <= start:
                        continue
                    firstBin = max(0, start // self.binSize)
                    lastBin = min(nr_bins, int(np.ceil(end / float(self.binSize))))
                    if lastBin > firstBin:
                        marked[firstBin:lastBin] = True
                        nr_regions += 1
        if nr_regions == 0:
            return None
        msg = "Chromosome {:s}: {:d} {:s} region(s) from BED cover {:d} of {:d} bins"
        print(msg.format(str(self.chromosome), nr_regions, pLabel,
                         int(marked.sum()), nr_bins))
        return marked

    def getExcludedBins(self):
        return self.__binsFromBed(self.excludedRegions, "excluded")

    def getIncludedBins(self):
        return self.__binsFromBed(self.includedRegions, "included")

    def __getNumberBins(self):
        if self.sparseHiCMatrix is not None:
            return self.sparseHiCMatrix.shape[0]
        if self.FactorDataArray is not None:
            return self.FactorDataArray.shape[0]
        if self.numBins is not None:
            return int(self.numBins)
        return 0

    def __computeSampleIndices(self):
        self.sampleIndices = None
        excluded = self.getExcludedBins()
        included = self.getIncludedBins()
        useCoverage = self.minTargetCoverage > 0.0
        if useCoverage and self.sparseHiCMatrix is None:
            msg = "minTargetCoverage is set but no Hi-C matrix is loaded; coverage filter ignored"
            log.warning(msg)
            useCoverage = False
        if not useCoverage and excluded is None and included is None:
            return
        nr_samples = self.__getNumberRawSamples()
        if not nr_samples:
            return
        flankingSize = self.flankingSize if self.flankingSize is not None else (0 if self.inputMode == "random2d" else self.windowSize)
        nr_bins = self.__getNumberBins()

        if self.windowMode == "grid" and self.gridCoordinates is not None:
            rStart = self.gridCoordinates[:, 0] + flankingSize
            cStart = self.gridCoordinates[:, 1] + flankingSize
            rStop = np.minimum(rStart + self.windowSize, nr_bins)
            cStop = np.minimum(cStart + self.windowSize, nr_bins)
            keep = np.ones(nr_samples, dtype=bool)
            nr_gaps = 0
            nr_excluded = 0
            nr_outside = 0
            if useCoverage:
                cumulative = np.concatenate(([0], np.cumsum(self.getBinCoverage())))
                covR = (cumulative[rStop] - cumulative[rStart]) / float(self.windowSize)
                covC = (cumulative[cStop] - cumulative[cStart]) / float(self.windowSize)
                coverageOK = (covR >= self.minTargetCoverage) & (covC >= self.minTargetCoverage)
                nr_gaps = int((~coverageOK).sum())
                keep &= coverageOK
            if excluded is not None:
                cumulativeExcl = np.concatenate(([0], np.cumsum(excluded)))
                untouched = ((cumulativeExcl[rStop] - cumulativeExcl[rStart]) == 0) & ((cumulativeExcl[cStop] - cumulativeExcl[cStart]) == 0)
                nr_excluded = int((~untouched).sum())
                keep &= untouched
            if included is not None:
                cumulativeIncl = np.concatenate(([0], np.cumsum(included)))
                inside = (((cumulativeIncl[rStop] - cumulativeIncl[rStart]) == (rStop - rStart)) &
                          ((cumulativeIncl[cStop] - cumulativeIncl[cStart]) == (cStop - cStart)))
                nr_outside = int((~inside).sum())
                keep &= inside
        else:
            startInd = np.arange(nr_samples) + flankingSize
            stopInd = np.minimum(startInd + self.windowSize, nr_bins)

            keep = np.ones(nr_samples, dtype=bool)
            nr_gaps = 0
            nr_excluded = 0
            nr_outside = 0
            if useCoverage:
                cumulative = np.concatenate(([0], np.cumsum(self.getBinCoverage())))
                coverage = (cumulative[stopInd] - cumulative[startInd]) / float(self.windowSize)
                coverageOK = coverage >= self.minTargetCoverage
                nr_gaps = int((~coverageOK).sum())
                keep &= coverageOK
            if excluded is not None:
                cumulativeExcl = np.concatenate(([0], np.cumsum(excluded)))
                untouched = (cumulativeExcl[stopInd] - cumulativeExcl[startInd]) == 0
                nr_excluded = int((~untouched).sum())
                keep &= untouched
            if included is not None:
                cumulativeIncl = np.concatenate(([0], np.cumsum(included)))
                inside = ((cumulativeIncl[stopInd] - cumulativeIncl[startInd])
                          == (stopInd - startInd))
                nr_outside = int((~inside).sum())
                keep &= inside

        self.sampleIndices = np.nonzero(keep)[0]
        msg = "Chromosome {:s}: keeping {:d} of {:d} samples ({:.1f} %)".format(
            str(self.chromosome), len(self.sampleIndices), nr_samples,
            100.0 * len(self.sampleIndices) / nr_samples)
        if useCoverage:
            msg += "; {:d} below target coverage {:.2f}".format(nr_gaps, self.minTargetCoverage)
        if excluded is not None:
            msg += "; {:d} overlapping an excluded region".format(nr_excluded)
        if included is not None:
            msg += "; {:d} not fully inside an included region".format(nr_outside)
        log.info(msg)
        print(msg)
        if len(self.sampleIndices) == 0:
            msg = ("Chromosome {:s} has no window left. Either it is not covered by the target "
                   "matrix, or minTargetCoverage ({:.2f}) is too strict for windowSize {:d}, or "
                   "the excluded regions span it entirely, or no included region is as long as "
                   "one window ({:d} bins = {:d} bp).").format(
                       str(self.chromosome), self.minTargetCoverage, self.windowSize,
                       self.windowSize, self.windowSize * self.binSize)
            log.warning(msg)

    def mapSampleIndex(self, idx):
        '''
        Translate a sample index (0 .. getNumberSamples()-1) into the window
        position it refers to. Identity unless gaps are being skipped.
        '''
        if self.sampleIndices is None:
            return idx
        return int(self.sampleIndices[idx])

    def getWindowCoordinates(self):
        """Return the effective list/array of (row_start, col_start) for grid mode, or 1D start indices for diagonal mode."""
        if self.windowMode == "grid" and self.gridCoordinates is not None:
            coords = self.gridCoordinates if self.sampleIndices is None else self.gridCoordinates[self.sampleIndices]
            return coords + int(self.flankingSize or 0)
        return self.sampleIndices

    def getNumberSamples(self):
        if not self.data_loaded:
            return None
        if self.sampleIndices is not None:
            return len(self.sampleIndices)
        return self.__getNumberRawSamples()

    def __getNumberRawSamples(self):
        if not self.data_loaded:
            return None
        if self.windowMode == "grid" and self.gridCoordinates is not None:
            n_grid = int(self.gridCoordinates.shape[0])
            print(f"Number of grid samples (chr{self.chromosome}): {n_grid}")
            return n_grid
        featureArrays = [self.FactorDataArray, self.sparseHiCMatrix, self.sequenceArray]
        cutouts = [self.windowSize+2*self.flankingSize, self.windowSize+2*self.flankingSize, (self.windowSize+2*self.flankingSize)*self.binSize]
        nr_samples_list = []
        for featureArray, cutout in zip(featureArrays, cutouts):
            if featureArray is not None:
                nr_samples_list.append(featureArray.shape[0] - cutout + 1)
            else:
                nr_samples_list.append(0)
        if not any(x > 0 for x in nr_samples_list):
            required_bins = self.windowSize + 2 * self.flankingSize
            nr_bins = self.FactorDataArray.shape[0] if self.FactorDataArray is not None else 0
            msg = ("Chromosome {:s} has only {:d} bins at binSize {:s}, but windowSize {:d} "
                   "(+2x flanking) needs at least {:d} bins to form one window. "
                   "Use a finer binSize (smaller -b) or a smaller windowSize, or drop this chromosome.").format(
                       str(self.chromosome), nr_bins, str(self.binSize), self.windowSize, required_bins)
            raise RuntimeError(msg)
        if len(set( [x for x in nr_samples_list if x>0] )) != 1:
            msg = "Error: sample binning / DNA sequence encoding went wrong"
            msg += " -- nr_samples_list: {:s}".format(str(nr_samples_list))
            msg += " -- featureArrays: {:s}".format(str([type(x) for x in featureArrays]))
            msg += " -- cutouts: {:s}".format(str(cutouts))
            raise RuntimeError(msg)
        print("Number of samples: " + " | ".join([str(x) for x in nr_samples_list]))
        return max(nr_samples_list)

    def __getMatrixData(self, idx):
        if self.matrixFilePath is None:
            return None
        if not self.data_loaded:
            msg = "Error: Load data first"
            raise RuntimeError(msg)
        windowSize = self.windowSize
        flankingSize = self.flankingSize
        if flankingSize is None:
            flankingSize = 0 if self.inputMode == "random2d" else windowSize
            self.flankingSize = flankingSize
        if self.windowMode == "grid" and self.gridCoordinates is not None:
            r0, c0 = int(self.gridCoordinates[idx, 0]), int(self.gridCoordinates[idx, 1])
            rStart, rStop = r0 + flankingSize, r0 + flankingSize + windowSize
            cStart, cStop = c0 + flankingSize, c0 + flankingSize + windowSize
            trainmatrix = self.sparseHiCMatrix[rStart:rStop, cStart:cStop].todense()
        else:
            startInd = idx + flankingSize
            stopInd = startInd + windowSize
            trainmatrix = self.sparseHiCMatrix[startInd:stopInd, startInd:stopInd].todense()
        trainmatrix = np.array(np.nan_to_num(trainmatrix))
        if self.targetValueRange is not None:
            lo, hi = self.targetValueRange[0], self.targetValueRange[1]
            log_tf = self.targetValueRange[2] if len(self.targetValueRange) > 2 else self.logTransformTarget
            trainmatrix = utils.toUnitRange(trainmatrix, lo, hi, pLogTransform=log_tf)
        trainmatrix = np.expand_dims(trainmatrix, axis=-1)
        return trainmatrix
    
    def __getFactorData(self, idx):
        if self.inputMode == "bigwig" and self.chromatinFolder is None:
            return None
        if not self.data_loaded:
            msg = "Error: Load data first"
            raise RuntimeError(msg)
        windowSize = self.windowSize
        flankingSize = self.flankingSize
        if flankingSize is None:
            flankingSize = 0 if self.inputMode == "random2d" else windowSize
            self.flankingSize = flankingSize

        if self.inputMode == "random2d":
            if self.windowMode == "grid" and self.gridCoordinates is not None:
                r0, c0 = int(self.gridCoordinates[idx, 0]), int(self.gridCoordinates[idx, 1])
                rStart, rStop = r0 + flankingSize, r0 + flankingSize + windowSize
                cStart, cStop = c0 + flankingSize, c0 + flankingSize + windowSize
                factorArray = self.RandomMatrix2D[rStart:rStop, cStart:cStop]
            else:
                startInd = idx + flankingSize
                stopInd = startInd + windowSize
                factorArray = self.RandomMatrix2D[startInd:stopInd, startInd:stopInd]
            factorArray = np.expand_dims(factorArray, axis=-1)
            return factorArray

        # 1D BigWig mode
        if self.windowMode == "grid" and self.gridCoordinates is not None:
            r0, c0 = int(self.gridCoordinates[idx, 0]), int(self.gridCoordinates[idx, 1])
            span = 2 * flankingSize + windowSize
            f_row = self.FactorDataArray[r0 : r0 + span]
            f_col = self.FactorDataArray[c0 : c0 + span]
            factorArray = np.concatenate([f_row, f_col], axis=-1)
        else:
            startIdx = idx
            endIdx = startIdx + 2 * flankingSize + windowSize
            factorArray = self.FactorDataArray[startIdx:endIdx]
        factorArray = np.expand_dims(factorArray, axis=-1)
        return factorArray

    def getSampleData(self, idx):
        if not self.data_loaded:
            return None
        idx = self.mapSampleIndex(idx)
        factorArray = self.__getFactorData(idx)
        matrixArray = self.__getMatrixData(idx)
        if matrixArray is not None:
            matrixArray = matrixArray.astype("float32")
        return {"factorData": factorArray.astype("float32"), 
                "out_matrixData": matrixArray}
        
    def plotFeatureAtIndex(self, idx, outpath, figuretype="png"):
        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to plot"
            print(msg)
            return
        if isinstance(idx, int) and (idx >= self.FactorDataArray.shape[0] or idx < 0):
            msg = "Error: Invalid index {:d}; must be None or integer in 0..{:d}".format(idx, self.FactorDataArray.shape[0]-1)
            raise ValueError(msg)
        if isinstance(idx, int):
            factorArray = self.__getFactorData(idx)
            startBin = idx
        else:
            factorArray = self.FactorDataArray 
            startBin = None
        for plotType in ["box", "line"]:   
            utils.plotChromatinFactors(pFactorArray=factorArray, 
                                        pFeatureNameList=self.factorNames,
                                        pChromatinFolder=self.chromatinFolder,
                                        pChrom=self.chromosome,
                                        pbinSize=self.binSize,
                                        pStartbin=startBin,
                                        pOutputPath=outpath,
                                        pPlotType=plotType,
                                        pFigureType=figuretype)
    
    def plotFeaturesAtPosition(self, position, outpath, figuretype="png"):
        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to plot"
            print(msg)
            return
        if isinstance(position, int) and position > self.chromSize_factors:
            msg = "Error: Invalid position {:d}; must be in 0..{:d}"
            msg = msg.format(position, self.chromSize_factors)
            raise ValueError(msg)
        #compute the bin index from the position
        elif isinstance(position, int):
            idx = int(np.floor(position / self.binSize))
        else:
            idx = None
        return self.plotFeatureAtIndex(idx=idx,
                                        outpath=outpath,
                                        figuretype=figuretype)

    def saveMatrix(self, outputpath, index=None):
        if not self.data_loaded:
            msg = "Warning: No data loaded, nothing to save"
            print(msg)
            return
        sparseMatrix = None
        windowSize = self.windowSize
        flankingSize = self.flankingSize
        if not isinstance(flankingSize, int):
            flankingSize = windowSize
        if isinstance(self.maximumDistance, int) and self.maximumDistance < windowSize and self.maximumDistance > 0:
            maximumDistance = self.maximumDistance
        else:
            maximumDistance = windowSize
        if isinstance(index, int) and index < self.getNumberSamples():
            tmpMat = np.zeros(shape=(windowSize, windowSize))
            indices = np.mask_indices(windowSize, utils.maskFunc, k=maximumDistance)
            tmpMat[indices] = self.__getMatrixData(idx=self.mapSampleIndex(index))
            sparseMatrix = csr_matrix(tmpMat)
        else:
            sparseMatrix = self.sparseHiCMatrix
        folderName = self.chromatinFolder.rstrip("/").replace("/","-")
        filename = "matrix_{:s}_chr{:s}_{:s}".format(folderName, str(self.chromosome), str(index))
        filename = os.path.join(outputpath, filename)
        save_npz(file=filename, matrix=sparseMatrix)

    def __prepareWriteoutDict(self, pFirstIndex, pLastIndex, pOutfolder):
        if not self.data_loaded:
            msg = "Error: no data loaded, nothing to prepare"
            raise RuntimeError(msg)

        def get_sample_data(idx):
            return self.getSampleData(idx=idx)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            data = list(executor.map(get_sample_data, range(pFirstIndex, pLastIndex)))
        recordDict = dict()
        storedFeaturesDict = dict()
        if len(data) < 1:
            msg = "Error: No data to write"
            raise RuntimeError(msg)

        for key in data[0]:
            featData = [feature[key] for feature in data]
            if not any(elem is None for elem in featData):
                recordDict[key] = np.array(featData)
                storedFeaturesDict[key] = {"shape": recordDict[key].shape[1:], "dtype": tfdtypes.as_dtype(recordDict[key].dtype)}

        return recordDict, storedFeaturesDict