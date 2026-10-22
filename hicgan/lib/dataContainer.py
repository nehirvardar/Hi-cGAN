import os
import numpy as np
from tensorflow import dtypes as tfdtypes
from scipy.sparse import save_npz, csr_matrix
from tqdm import tqdm
import concurrent.futures
import scipy.sparse as sp
from . import utils
from . import records
import cooler


import logging
log = logging.getLogger(__name__)


class DataContainer():
    def __init__(self, chromosome, matrixFilePath, chromatinFolder, binSize=None):
        self.chromosome = str(chromosome)
        self.matrixFilePath = matrixFilePath
        self.chromatinFolder = chromatinFolder
        self.FactorDataArray = None
        self.RandomInputArray = None
        self.nr_factors = None
        self.sparseHiCMatrix = None
        self.sequenceArray = None
        self.binSize = None
        if matrixFilePath is None: #otherwise it will be defined by the Hi-C matrix itself upon loading
            self.binSize = binSize
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
        self.minTargetCoverage = 0.0
        self.excludedRegions = None
        self.includedRegions = None
        self.targetValueRange = None
        self.sampleIndices = None
        self.data_loaded = False

    def __loadFactorData(self, ignoreChromLengths=False, scaleFeatures=False, clampFeatures=False):
        if self.chromatinFolder is None:
            return

        import glob
        import os
        import numpy as np
        import scipy.sparse as sp

        c_clean = str(self.chromosome).replace("chr", "")
        candidates = [
            os.path.join(self.chromatinFolder, f"chr{c_clean}.npz"),
            os.path.join(self.chromatinFolder, f"{c_clean}.npz"),
        ]
        npz_files = [f for f in candidates if os.path.exists(f)]
        if len(npz_files) == 0:
            search_pattern = os.path.join(self.chromatinFolder, f"*{self.chromosome}.npz")
            npz_files = glob.glob(search_pattern)

        if len(npz_files) == 0:
            print(f"In {self.chromatinFolder} no .npz file for {self.chromosome} ")
            return

        npz_file = npz_files[0]

        try:
            # Load sparse RNA interaction matrix at original 250kb resolution (no 250kb -> 25kb upscaling)
            sparse_mat = sp.load_npz(npz_file)
            factor_matrix = sparse_mat.toarray().astype(np.float32)  # (N_rna, N_rna) 2D dense

            if scaleFeatures:
                m_max = factor_matrix.max()
                if m_max > 0:
                    factor_matrix /= m_max

            self.FactorDataArray = factor_matrix

            # Load the pre-saved fixed random matrix for this chromosome from data/random_input/
            parent_dir = os.path.dirname(os.path.normpath(self.chromatinFolder))
            random_dir = os.path.join(parent_dir, "random_input")
            random_file = os.path.join(random_dir, f"chr{c_clean}.npy")

            if not os.path.exists(random_file):
                raise FileNotFoundError(
                    f"Pre-generated random matrix not found at {random_file}. "
                    f"Expected fixed random matrices in {random_dir}."
                )

            self.RandomInputArray = np.load(random_file).astype(np.float32)

            if self.binSize is None:
                self.binSize = 250000  # Default 250kb resolution for RNA matrices

            self.chromSize_factors = self.chromSize_matrix if self.chromSize_matrix else factor_matrix.shape[0] * self.binSize
            self.nr_factors = 1
            self.factorNames = [os.path.basename(npz_file)]

            print(f"[{self.chromosome}] Loaded RNA GT {self.FactorDataArray.shape} and Random Input ({random_file})")

        except Exception as e:
            raise IOError(f"Error while loading RNA data: {e}")


    def __loadMatrixData(self, scaleMatrix=False):
        if self.matrixFilePath is None:
            return
        try:
            chromsizes = utils.getChromSizesFromCooler(self.matrixFilePath)
            actual_chrom = "chr" + self.chromosome if ("chr" + self.chromosome) in chromsizes else self.chromosome
            if actual_chrom not in chromsizes:
                actual_chrom = self.chromosome.replace("chr", "")

            chromsize_matrix = chromsizes[actual_chrom]

            c = cooler.Cooler(self.matrixFilePath)
            binSize = c.binsize
            sparseHiCMatrix = c.matrix(balance=False, sparse=True).fetch(actual_chrom).tocsr()

        except Exception as e:
            msg = f"Error: Could not load data from Hi-C matrix {self.matrixFilePath} Error details: {e}"
            raise IOError(msg)


        #scale to 0..1, if requested
        if scaleMatrix:
            sparseHiCMatrix = utils.scaleArray(sparseHiCMatrix)

        #ensure that chrom sizes for matrix and factors are the same
        if self.chromSize_factors is not None and self.chromSize_factors != chromsize_matrix:
            msg = "Chromsize of matrix does not match bigwig files\n"
            msg += "Matrix: {:d} -- Bigwig files: {:d}"
            msg = msg.format(chromsize_matrix, self.chromSize_factors)
            raise IOError(msg)
        self.chromSize_matrix = chromsize_matrix

        #ensure that binSizes for matrix and factors match
        if self.binSize is None or self.binSize == binSize:
            self.binSize = binSize
            self.sparseHiCMatrix = sparseHiCMatrix
        elif self.binSize is not None and self.binSize != binSize:
            msg = "Matrix has wrong binSize\n"
            msg += "Matrix: {:d} -- Binned chromatin factors {:d}"
            msg = msg.format(binSize, self.binSize)
            raise IOError(msg)
        msg = "Loaded cooler matrix {:s}\n".format(self.matrixFilePath)
        msg += "chr. {:s}, matshape {:d}*{:d} -- min. {:d} -- max. {:d} -- nnz. {:d}"
        msg = msg.format(self.chromosome, self.sparseHiCMatrix.shape[0], self.sparseHiCMatrix.shape[1], int(self.sparseHiCMatrix.min()), int(self.sparseHiCMatrix.max()), self.sparseHiCMatrix.getnnz() )
        print(msg)
    
    def __unloadFactorData(self):
        #unload chromatin factor data to save memory, but do not touch metadata 
        self.FactorDataArray = None
        self.RandomInputArray = None
        
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
        self.excludedRegions = None
        self.includedRegions = None
        self.targetValueRange = None
        self.data_loaded = False

    def loadData(self, windowSize, flankingSize=None, maximumDistance=None, scaleFeatures=False, clampFeatures=False, scaleTargets=False, minTargetCoverage=0.0, excludedRegions=None, includedRegions=None, targetValueRange=None):
        if not isinstance(windowSize, int):
            msg = "windowSize must be integer"
            raise TypeError(msg)
        if isinstance(maximumDistance, int):
            maximumDistance = np.clip(maximumDistance, a_min=1, a_max=self.windowSize)

        # Load RNA matrix data directly (used as Ground Truth, no .cool Hi-C matrix needed)
        self.__loadFactorData(scaleFeatures=scaleFeatures, clampFeatures=clampFeatures)
        self.windowSize = windowSize
        self.flankingSize = flankingSize if flankingSize is not None else 0
        self.maximumDistance = maximumDistance
        self.minTargetCoverage = float(minTargetCoverage)
        self.excludedRegions = excludedRegions
        self.includedRegions = includedRegions
        self.targetValueRange = targetValueRange
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
        log.debug("Factors: {:s} -- Matrix: {:s} -- windowSize: {:s} -- flankingSize: {:s} -- maximumDistance: {:s}".format(str(factorsOK), str(matrixOK), str(windowSizeOK), str(flankingSizeOK), str(maximumDistanceOK)))
        log.debug("Chromatin folder: {:s} -- Nr. factors: {:s}".format(str(self.chromatinFolder), str(self.nr_factors)))
        log.debug("Chromatin folder: {:s} -- Nr. factors: {:s}".format(str(container.chromatinFolder), str(container.nr_factors)))
        #sanity check loading of bigwig files
        if self.chromatinFolder is not None and self.nr_factors is None:
            return False
        if container.chromatinFolder is not None and container.nr_factors is None:
            return False
        #if chromatin factors are present, the numbers and names of chromatin factors must match
        factorsOK = factorsOK and (self.nr_factors == container.nr_factors)
        # factorsOK = factorsOK and (self.factorNames == container.factorNames)
        log.debug("self.nr_factors: {:d} -- container.nr_factors: {:d}".format(self.nr_factors, container.nr_factors))
        log.debug("self.factorNames: {:s} -- container.factorNames: {:s}".format(str(self.factorNames), str(container.factorNames)))
        return factorsOK and matrixOK and windowSizeOK and flankingSizeOK and maximumDistanceOK
        
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
        # Write single TFRecord files using a clean folder basename prefix
        folderName = os.path.basename(os.path.normpath(self.chromatinFolder))
        recordfiles = [os.path.join(pOutputFolder, "data_{:s}_{:s}_{:03d}.tfrecord".format(folderName, str(self.chromosome), i + 1)) for i in range(nr_files)]

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
        windowSize/flankingSize. The sliding window spans
        windowSize + 2*flankingSize bins, so chromosomes that are short
        relative to the chosen resolution (large binSize) yield no samples.
        '''
        if not self.data_loaded or self.FactorDataArray is None:
            return False
        required_bins = self.windowSize + 2 * self.flankingSize
        return self.FactorDataArray.shape[0] >= required_bins

    def getBinCoverage(self):
        '''
        Boolean array, one entry per bin of the target matrix, True where the bin
        has at least one contact. Bins inside a gap of the matrix have an all-zero
        row and come out False. Returns None if no matrix is loaded.
        '''
        if self.sparseHiCMatrix is not None:
            rowNnz = np.asarray((self.sparseHiCMatrix != 0).sum(axis=1)).ravel()
            return rowNnz > 0
        if self.FactorDataArray is not None:
            rowNnz = np.asarray((self.FactorDataArray != 0).sum(axis=1)).ravel()
            return rowNnz > 0
        return None

    def __binsFromBed(self, pPathList, pLabel):
        '''
        Boolean array, one entry per bin, True where the bin overlaps a region
        listed in the given BED file(s). Returns None if no file was given or
        none of its regions falls on this chromosome.

        Both naming conventions are accepted in the BED file, "chr21" and "21",
        regardless of which one the matrix and the bigwig files use.

        Shared by getExcludedBins and getIncludedBins: the two differ only in
        what the caller does with the result, so parsing lived in one of them
        and was about to be copied into the other.
        '''
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
                    #BED is half-open and 0-based, so a region ending exactly on a
                    #bin boundary must not mark the following bin
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
        '''
        Bins overlapping a region of the excludedRegions BED file(s).
        '''
        return self.__binsFromBed(self.excludedRegions, "excluded")

    def getIncludedBins(self):
        '''
        Bins overlapping a region of the includedRegions BED file(s).

        This is the counterpart used at PREDICTION time: it restricts the
        sliding window to a named set of loci instead of running over the whole
        chromosome. The intended use is to predict exactly the regions that were
        held out of training, so nothing the model was fitted on is scored.
        '''
        return self.__binsFromBed(self.includedRegions, "included")

    def __getNumberBins(self):
        if self.sparseHiCMatrix is not None:
            return self.sparseHiCMatrix.shape[0]
        if self.FactorDataArray is not None:
            return self.FactorDataArray.shape[0]
        return 0

    def __computeSampleIndices(self):
        '''
        Decide which sliding-window positions are used as samples.

        By default every position is used, self.sampleIndices stays None and
        nothing about the container changes. Two filters can restrict the set,
        and they combine:

        1. minTargetCoverage > 0 drops the positions whose TARGET submatrix lies
           in a gap of the Hi-C matrix, instead of handing them to the model as
           a block of zeros. This matters because __getMatrixData() runs the
           target through nan_to_num, so a missing region is indistinguishable
           from a region of genuinely zero contacts: the model is otherwise
           taught to predict an empty map wherever the data happen to be absent.
           On Akita's published GM12878 train fold, for instance, 25 % of the
           window positions on the training chromosomes fall into such gaps.

           Coverage is measured per bin rather than per pixel (see
           getBinCoverage); gaps are contiguous stretches of the matrix with no
           contacts at all, so the two agree, and the per-bin form is O(n)
           instead of O(n * windowSize^2).

        2. excludedRegions drops every position whose target window overlaps a
           region listed in a BED file. Use it to hold regions out of training
           deliberately: another method's test set, a blacklist, a locus kept
           for evaluation. The overlap is tested against the target window only.
           Chromatin features from the flanks may still reach into an excluded
           region, which is intended: no label from that region is ever used,
           but the model still sees the context around the windows it keeps.

        3. includedRegions is the mirror image, used at prediction time: keep
           ONLY the positions whose target window lies ENTIRELY inside a region
           listed in a BED file. Containment, not overlap, and deliberately so.
           It makes filter 3 the exact complement of filter 2: a window that
           training dropped for touching an excluded region is a window that
           overlaps it, and the windows fully inside are a subset of those. So
           predicting with includedRegions = the same BED that training was
           given as excludedRegions cannot produce a window containing a single
           bin the model was fitted on.
        '''
        self.sampleIndices = None
        excluded = self.getExcludedBins()
        included = self.getIncludedBins()
        useCoverage = self.minTargetCoverage > 0.0
        if useCoverage and self.sparseHiCMatrix is None and self.FactorDataArray is None:
            msg = "minTargetCoverage is set but no target matrix is loaded; coverage filter ignored"
            log.warning(msg)
            useCoverage = False
        if not useCoverage and excluded is None and included is None:
            return
        nr_samples = self.__getNumberRawSamples()
        if not nr_samples:
            return
        flankingSize = self.flankingSize if self.flankingSize is not None else 0
        nr_bins = self.__getNumberBins()
        startInd = np.arange(nr_samples) + flankingSize
        #windows running past the end of the matrix (the factor array can be
        #longer than the matrix) count the missing bins as uncovered
        stopInd = np.minimum(startInd + self.windowSize, nr_bins)

        keep = np.ones(nr_samples, dtype=bool)
        nr_gaps = 0
        nr_excluded = 0
        nr_outside = 0
        if useCoverage:
            #cumulative sum gives the covered-bin count of every window in one pass
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
            #fully contained: every bin of the target window is marked. The
            #window length is stopInd - startInd rather than self.windowSize,
            #because a window running past the end of the matrix is shorter and
            #comparing against the nominal size would reject all of them.
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

    def getNumberSamples(self):
        if not self.data_loaded:
            return None
        if self.sampleIndices is not None:
            return len(self.sampleIndices)
        return self.__getNumberRawSamples()

    def __getNumberRawSamples(self):
        if not self.data_loaded:
            return None
        featureArrays = [self.FactorDataArray, self.sparseHiCMatrix, self.sequenceArray]
        cutouts = [self.windowSize+2*self.flankingSize, self.windowSize+2*self.flankingSize, (self.windowSize+2*self.flankingSize)*self.binSize]
        nr_samples_list = []
        for featureArray, cutout in zip(featureArrays, cutouts):
            if featureArray is not None:
                nr_samples_list.append(featureArray.shape[0] - cutout + 1)
            else:
                nr_samples_list.append(0)
        #if the (only) available feature is too short for one window, give an
        #actionable message instead of the cryptic "binning went wrong" below.
        #This is the typical failure when predicting at a coarse binSize.
        if not any(x > 0 for x in nr_samples_list):
            required_bins = self.windowSize + 2 * self.flankingSize
            nr_bins = self.FactorDataArray.shape[0] if self.FactorDataArray is not None else 0
            msg = ("Chromosome {:s} has only {:d} bins at binSize {:s}, but windowSize {:d} "
                   "(+2x flanking) needs at least {:d} bins to form one window. "
                   "Use a finer binSize (smaller -b) or a smaller windowSize, or drop this chromosome.").format(
                       str(self.chromosome), nr_bins, str(self.binSize), self.windowSize, required_bins)
            raise RuntimeError(msg)
        #check if all features have the same number of samples
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
            return None # this can't work
        if not self.data_loaded:
            msg = "Error: Load data first"
            raise RuntimeError(msg)
        #the 0-th matrix starts flankingSize away from the boundary
        windowSize = self.windowSize
        flankingSize = self.flankingSize
        if flankingSize is None:
            flankingSize = windowSize
            self.flankingSize = windowSize
        startInd = idx + flankingSize
        stopInd = startInd + windowSize
        trainmatrix = self.sparseHiCMatrix[startInd:stopInd,startInd:stopInd].todense()
        trainmatrix = np.array(np.nan_to_num(trainmatrix))
        if self.targetValueRange is not None:
            #Map the target onto [0, 1], the range the denoiser's sigmoid can
            #actually produce. The bounds come from the data (see
            #utils.observedValueRange), not from a constant, and the SAME bounds
            #are used for every chromosome and inverted at prediction time.
            #
            #Without this the target keeps its native range. For an Akita-style
            #target that is roughly [-2, +2] with 56 % of a window below zero,
            #and a sigmoid cannot emit a negative number at all: the pixel loss
            #then carries an offset it can never remove, and the discriminator
            #can separate real from generated on sign alone, which makes the
            #adversarial gradient useless.
            lo, hi = self.targetValueRange
            trainmatrix = utils.toUnitRange(trainmatrix, lo, hi)
        trainmatrix = np.expand_dims(trainmatrix, axis=-1) #make Hi-C (sub-)matrix an RGB image
        return trainmatrix

    def __getFactorData(self, idx):
        if self.chromatinFolder is None or self.FactorDataArray is None:
            return None
        if not self.data_loaded:
            msg = "Error: Load data first"
            raise RuntimeError(msg)

        windowSize = self.windowSize
        flankingSize = self.flankingSize
        if flankingSize is None:
            flankingSize = 0
            self.flankingSize = 0

        startInd = idx + flankingSize
        stopInd = startInd + windowSize

        factorArray = self.FactorDataArray[startInd:stopInd, startInd:stopInd]
        factorArray = np.nan_to_num(factorArray)

        if self.targetValueRange is not None:
            lo, hi = self.targetValueRange
            factorArray = utils.toUnitRange(factorArray, lo, hi)

        factorArray = np.expand_dims(factorArray, axis=-1)
        return factorArray

    def __getRandomInputData(self, idx):
        if self.RandomInputArray is None:
            return None
        windowSize = self.windowSize
        flankingSize = self.flankingSize if self.flankingSize is not None else 0
        startInd = idx + flankingSize
        stopInd = startInd + windowSize

        # Slice the 2D window from the chromosome's fixed random matrix
        randomWindow = self.RandomInputArray[startInd:stopInd, startInd:stopInd]
        randomWindow = np.expand_dims(randomWindow, axis=-1)
        return randomWindow

    def getSampleData(self, idx):
        if not self.data_loaded:
            return None
        idx = self.mapSampleIndex(idx)
        rna_gt_matrix = self.__getFactorData(idx).astype("float32")
        random_input = self.__getRandomInputData(idx).astype("float32")

        return {
            "factorData": random_input,      # Window sliced from the chromosome's single random matrix
            "out_matrixData": rna_gt_matrix  # Window sliced from the chromosome's Ground Truth RNA matrix
        }
        
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