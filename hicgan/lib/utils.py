#!python3
import os
import cooler
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
from scipy import sparse
from sklearn import metrics as metrics
import traceback


import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix, issparse
import logging
log = logging.getLogger(__name__)


def getBigwigFileList(pDirectory):
    #returns a list of bigwig files in pDirectory
    retList = []
    for file in sorted(os.listdir(pDirectory)):
        if file.endswith(".bigwig") or file.endswith("bigWig") or file.endswith(".bw"):
            retList.append(pDirectory + file)
    return retList
         
def getMatrixFromCooler(pCoolerFilePath, pChromNameStr):
    #returns sparse csr matrix from cooler file for given chromosome name
    sparseMatrix = None
    binSizeInt = 0
    try:
        coolerMatrix = cooler.Cooler(pCoolerFilePath)
        sparseMatrix = coolerMatrix.matrix(sparse=True,balance=False).fetch(pChromNameStr)
        sparseMatrix = sparseMatrix.tocsr() #so it can be sliced later
        binSizeInt = int(coolerMatrix.binsize)
    except Exception as e:
        traceback.print_exc()
        print(coolerMatrix.chromnames)
        print(e)
    return sparseMatrix, binSizeInt

def getChromSizesFromCooler(pCoolerFilePath):
    #get the sizes of the chromosomes present in a cooler matrix
    chromSizes = dict()
    try:
        coolerMatrix = cooler.Cooler(pCoolerFilePath) 
        chromSizes = coolerMatrix.chromsizes.to_dict()
    except Exception as e:
        print(e)
    return chromSizes


def scaleArray(pArray):
    '''
    min-max scaling for numpy arrays and sparse csr matrices

    Parameters:
    pArray (np.ndarray or sparse.csr_matrix): array to scale

    Returns:
    array scaled to value range [0..1]
    ''' 
    if pArray is None or pArray.size == 0:
        msg = "cannot normalize empty array"
        print(msg)
        return pArray
    #A sparse matrix holding negative values cannot be shifted: scipy refuses to
    #subtract a nonzero scalar because it would have to materialise every
    #implicit zero. Say so here rather than letting the caller meet scipy's
    #message, which does not hint at the cause or the cure.
    if sparse.issparse(pArray) and pArray.min() < 0:
        raise NotImplementedError(
            "scaleArray cannot min-max a SPARSE matrix whose values go negative "
            "(min {:.4g}). An Akita-style log observed/expected target is about "
            "half negative. Use hicTraining --scaleTargetToUnitRange instead: it "
            "measures the range once, maps the dense target window onto [0, 1], "
            "and records the range so hicPredict --targetValueRange can invert "
            "it.".format(float(pArray.min())))
    if pArray.max() - pArray.min() != 0:
        normArray = (pArray - pArray.min()) / (pArray.max() - pArray.min())
    elif pArray.max() > 0: #min = max >0
        normArray = pArray / pArray.max()
    else: #min=max <= 0
        normArray = np.zeros_like(pArray)
    return normArray

def observedValueRange(pMatrixFilePaths, pChromosomes=None):
    """Smallest and largest value actually present in the target matrices.

    Measured, not assumed. The range is taken over the STORED entries only: an
    absent entry in a sparse Hi-C matrix means "not observed", not "value zero",
    and letting it into the range would drag the minimum to 0 for any target
    whose real values are negative.

    One range is computed across every training matrix and chromosome, and the
    same one is then used everywhere. A per-chromosome range would map the same
    physical value to a different target on chr1 than on chr2, so the model
    would be chasing a target that moves between chromosomes.
    """
    vmin, vmax = np.inf, -np.inf
    for path in ([pMatrixFilePaths] if isinstance(pMatrixFilePaths, str)
                 else pMatrixFilePaths):
        prefix = getChromPrefixCooler(path)
        chroms = pChromosomes
        if chroms is None:
            chroms = [c.replace(prefix, "", 1) for c in getChromSizesFromCooler(path)]
        for chrom in chroms:
            name = prefix + str(chrom).replace(prefix, "", 1)
            try:
                M, _ = getMatrixFromCooler(path, name)
            except Exception:
                continue
            if M is None:
                continue
            data = M.data if sparse.issparse(M) else np.asarray(M).ravel()
            data = data[np.isfinite(data)]
            if data.size == 0:
                continue
            vmin = min(vmin, float(data.min()))
            vmax = max(vmax, float(data.max()))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        msg = "could not determine a usable value range from the target matrices"
        raise ValueError(msg)
    return vmin, vmax


def toUnitRange(pArray, pMin, pMax):
    """Map a DENSE array from [pMin, pMax] onto [0, 1].

    Dense on purpose. The same map on a sparse matrix would have to shift the
    implicit zeros too, which scipy refuses ("subtracting a nonzero scalar from
    a sparse array is not supported") -- the error that made the old
    scaleArray() path unusable for any signed target. Target windows are
    densified before use anyway, so the transform is applied there.
    """
    span = float(pMax) - float(pMin)
    if span <= 0:
        return np.zeros_like(pArray, dtype=float)
    return np.clip((np.asarray(pArray, dtype=float) - float(pMin)) / span, 0.0, 1.0)


def fromUnitRange(pMatrix, pMin, pMax):
    """Inverse of toUnitRange: [0, 1] back to [pMin, pMax], the real units.

    On a sparse matrix only the STORED values are mapped. That is deliberate:
    an unstored pixel was never predicted, and giving it pMin would invent a
    maximally depleted contact everywhere the model said nothing.
    """
    span = float(pMax) - float(pMin)
    if sparse.issparse(pMatrix):
        out = pMatrix.copy().astype(np.float64)
        out.data = out.data * span + float(pMin)
        return out
    return np.asarray(pMatrix, dtype=float) * span + float(pMin)


def saveValueRange(pPath, pMin, pMax):
    import json
    with open(pPath, "w") as fh:
        json.dump({"min": float(pMin), "max": float(pMax)}, fh, indent=1)
    log.info("wrote target value range [%g, %g] to %s", pMin, pMax, pPath)


def loadValueRange(pPath):
    import json
    with open(pPath) as fh:
        d = json.load(fh)
    return float(d["min"]), float(d["max"])


def showMatrix(pMatrix):
    #test function to show matrices
    #debug only, not for production use
    # print(pMatrix.max())
    plotmatrix = pMatrix + 1
    plt.matshow(plotmatrix, cmap="Reds", norm=colors.LogNorm())
    plt.show()

def plotMatrix(pMatrix, pFilename, pTitle):
    '''
    helper function to plot dense numpy 2D matrices in logscale to a file
    
    Parameters:
    pMatrix (numpy.ndarray): The matrix to plot, must be 2D
    pFilename (str): The filename for the plot, should have file extension .png, .pdf or .svg
    pTitle (str): A title that will appear on the plot

    Returns:
    None
    '''
    if not isinstance(pMatrix, np.ndarray) \
            or len(pMatrix.shape) != 2:
        return
    fig1, ax1 = plt.subplots()
    cs = ax1.matshow(pMatrix, cmap="RdYlBu_r", norm=colors.LogNorm())
    ax1.set_title(str(pTitle))
    fig1.colorbar(cs)
    fig1.savefig(pFilename)
    plt.close(fig1)
    del fig1, ax1

def plotLoss(pGeneratorLossValueLists, pDiscLossValueLists, pGeneratorLossNameList, pDiscLossNameList, pFilename, useLogscaleList=[False, False]):
    #plot loss and validation loss over epoch numbers
    fig1, ax1 = plt.subplots(figsize=(6,4.5))
    nr_epochs = len(pGeneratorLossValueLists[0])
    x_vals = np.arange(nr_epochs) + 1
    for generatorLossVals, _ in zip(pGeneratorLossValueLists, pGeneratorLossNameList):
        ax1.plot(x_vals, generatorLossVals)
    ax1.set_title('model loss')
    ax1.set_ylabel('generator loss')
    ax1.set_xlabel('epoch')
    if useLogscaleList[0]:
        ax1.set_yscale('log')
    ax2 = ax1.twinx()
    for discLossVals, _ in zip(pDiscLossValueLists, pDiscLossNameList):
        ax2.plot(x_vals, discLossVals, ":")
    ax2.set_ylabel("discriminator loss")
    if useLogscaleList[1]:
        ax2.set_yscale('log')
    locVal = 0
    if nr_epochs <= 25:
        locVal = 1
    elif nr_epochs <= 50:
        locVal = 5
    elif nr_epochs <= 100:
        locVal = 10
    elif nr_epochs <= 500:
        locVal = 50
    elif nr_epochs <= 1000:
        locVal = 100
    elif nr_epochs <= 3000:
        locVal = 500
    elif nr_epochs <= 5000:
        locVal = 600
    else:
        locVal = 1000
    ax1.xaxis.set_major_locator(MultipleLocator(locVal))
    ax1.grid(True, which="both")
    if len(pGeneratorLossNameList) > 1:
        ax1.legend(pGeneratorLossNameList, loc='upper right', title="Generator")
    if len(pDiscLossNameList) > 1:
        ax2.legend(pDiscLossNameList, loc="lower right", title="Discriminator")
    fig1.tight_layout()
    fig1.savefig(pFilename)
    plt.close(fig1)
    del fig1, ax1, ax2

# def rebuildMatrix(pArrayOfTriangles, pWindowSize, pFlankingSize=None, pMaxDist=None, pStepsize=1):
#     #rebuilds the interaction matrix (a trapezoid along its diagonal)
#     #by taking the mean of all overlapping triangles
#     #returns an interaction matrix as a numpy ndarray
#     # if pFlankingSize == None:
#     #     flankingSize = pWindowSize
#     # else:
#     #     flankingSize = pFlankingSize
#     # nr_matrices = pArrayOfTriangles.shape[0]
#     # sum_matrix = np.zeros( (nr_matrices - 1 + (pWindowSize+2*flankingSize), nr_matrices - 1 + (pWindowSize+2*flankingSize)) )
#     # count_matrix = np.zeros_like(sum_matrix,dtype=int)    
#     # mean_matrix = np.zeros_like(sum_matrix,dtype="float32")
#     # if pMaxDist is None or pMaxDist == pWindowSize:
#     #     stepsize = 1
#     # else:
#     #     #trapezoid, compute the stepsize such that the overlap is minimized
#     #     stepsize = max(pStepsize, 1)
#     #     stepsize = min(stepsize, pWindowSize - pMaxDist + 1) #the largest possible value such that predictions are available for all bins
#     # #sum up all the triangular or trapezoidal matrices, shifting by one along the diag. for each matrix
#     # for i in tqdm(range(0, nr_matrices, stepsize), desc="rebuilding matrix"):
#     #     j = i + flankingSize
#     #     k = j + pWindowSize
#     #     if pMaxDist is None or pMaxDist == pWindowSize: #triangles
#     #         sum_matrix[j:k,j:k][np.triu_indices(pWindowSize)] += pArrayOfTriangles[i]
#     #     else: #trapezoids
#     #         sum_matrix[j:k,j:k][np.mask_indices(pWindowSize, maskFunc, pMaxDist)] += pArrayOfTriangles[i]
#     #     count_matrix[j:k,j:k] += np.ones((pWindowSize,pWindowSize),dtype=int) #keep track of how many matrices have contributed to each position
#     # mean_matrix[count_matrix!=0] = sum_matrix[count_matrix!=0] / count_matrix[count_matrix!=0]
#     # return mean_matrix

#     log.info("rebuilding matrix from triangles with window size {:d} and flanking size {:d}".format(pWindowSize, pFlankingSize if pFlankingSize is not None else 0))
#     if pFlankingSize == None:
#         flankingSize = pWindowSize
#     else:
#         flankingSize = pFlankingSize
#     nr_matrices = pArrayOfTriangles.shape[0]
#     log.info(f"Number of triangles: {nr_matrices}, window size: {pWindowSize}, flanking size: {flankingSize}")
    
#     matrix_size = nr_matrices - 1 + (pWindowSize+2*flankingSize)
#     sum_matrix = sparse.lil_matrix((matrix_size, matrix_size), dtype=np.float64)
#     log.debug(f"sum_matrix shape: {sum_matrix.shape}")
#     count_matrix = sparse.lil_matrix((matrix_size, matrix_size), dtype=np.int32)
#     log.debug(f"count_matrix shape: {count_matrix.shape}")
#     mean_matrix = sparse.lil_matrix((matrix_size, matrix_size), dtype=np.float32)
#     log.debug(f"mean_matrix shape: {mean_matrix.shape}")
    
#     if pMaxDist is None or pMaxDist == pWindowSize:
#         stepsize = 1
#     else:
#         #trapezoid, compute the stepsize such that the overlap is minimized
#         stepsize = max(pStepsize, 1)
#         stepsize = min(stepsize, pWindowSize - pMaxDist + 1) #the largest possible value such that predictions are available for all bins
#     #sum up all the triangular or trapezoidal matrices, shifting by one along the diag. for each matrix
#     log.info(f"Starting matrix rebuilding: {nr_matrices} triangles, window size {pWindowSize}, flanking size {flankingSize}, stepsize {stepsize}")
#     for i in tqdm(range(0, nr_matrices, stepsize), desc="rebuilding matrix"):
#         j = i + flankingSize
#         k = j + pWindowSize
#         if pMaxDist is None or pMaxDist == pWindowSize: #triangles
#             triu_indices = np.triu_indices(pWindowSize)
#             rows = triu_indices[0] + j
#             cols = triu_indices[1] + j
#             # Add values element by element for sparse matrices
#             for row, col, val in zip(rows, cols, pArrayOfTriangles[i]):
#                 sum_matrix[row, col] += val
#                 count_matrix[row, col] += 1
#         else: #trapezoids
#             mask_indices = np.mask_indices(pWindowSize, maskFunc, pMaxDist)
#             rows = mask_indices[0] + j
#             cols = mask_indices[1] + j
#             # Add values element by element for sparse matrices
#             for row, col, val in zip(rows, cols, pArrayOfTriangles[i]):
#                 sum_matrix[row, col] += val
#                 count_matrix[row, col] += 1
    
#     # Convert to CSR for efficient division
#     sum_csr = sum_matrix.tocsr()
#     count_csr = count_matrix.tocsr()
    
#     # Find non-zero elements in count matrix
#     nonzero_mask = count_csr.data != 0
    
#     # Create mean matrix by dividing sum by count where count != 0
#     mean_csr = sum_csr.copy().astype(np.float32)
#     mean_csr.data[nonzero_mask] = sum_csr.data[nonzero_mask] / count_csr.data[nonzero_mask]
    
#     # Convert back to dense numpy array to maintain original return type
#     mean_matrix = mean_csr.toarray()
    
#     log.info("Matrix rebuilding complete. Mean matrix shape: %s", mean_matrix.shape)
#     return mean_matrix

def bandToCsr(pBandSum, pStartCount, pWindowSize, pMatrixSize):
    """Turn an accumulated band into the mean matrix, as CSR.

    pBandSum holds, for every cell of the rebuilt matrix, the sum of the
    contributions of all windows covering it, laid out as (pWindowSize,
    pMatrixSize) indexed by [offset-from-diagonal, row]. pStartCount is how many
    windows start at each position.

    The divisor is never stored. How many windows cover a cell follows from the
    start positions alone: a cell on diagonal d at row p is covered by the
    windows starting in [p - (W-1-d), p], which one prefix sum answers for every
    cell at once.
    """
    W = int(pWindowSize)
    matrix_size = int(pMatrixSize)
    band_sum = np.asarray(pBandSum).reshape(W, matrix_size)
    cum = np.zeros(matrix_size + 1, dtype=np.int64)
    np.cumsum(np.asarray(pStartCount, dtype=np.int64), out=cum[1:])

    def _coverage(pDiagonal):
        """How many windows cover each row of diagonal pDiagonal."""
        k = W - 1 - pDiagonal
        if k == 0:
            shifted = cum[:matrix_size]
        else:
            shifted = np.concatenate((np.zeros(k, dtype=np.int64),
                                      cum[:matrix_size - k]))
        return cum[1:] - shifted

    # The CSR is filled directly rather than going through COO. A COO built
    # from the band would hold a second full copy of the values plus two
    # index arrays, and .tocsr() would then allocate a third. Here the only
    # arrays that ever exist are the band and the CSR itself. Within a row
    # the diagonals are visited in increasing order, which is exactly CSR's
    # required column order, so nothing has to be sorted either.
    nnzPerRow = np.zeros(matrix_size, dtype=np.int32)
    for dIdx in range(W):
        nnzPerRow += (_coverage(dIdx) > 0) & (band_sum[dIdx] != 0)
    indptr = np.zeros(matrix_size + 1, dtype=np.int64)
    np.cumsum(nnzPerRow, out=indptr[1:])
    del nnzPerRow
    total = int(indptr[-1])
    data = np.empty(total, dtype=np.float32)
    indices = np.empty(total, dtype=np.int32)
    running = np.zeros(matrix_size, dtype=np.int64)
    for dIdx in range(W):
        cov = _coverage(dIdx)
        rowsHere = np.flatnonzero((cov > 0) & (band_sum[dIdx] != 0))
        if rowsHere.size == 0:
            continue
        at = indptr[rowsHere] + running[rowsHere]
        data[at] = band_sum[dIdx, rowsHere] / cov[rowsHere]
        indices[at] = rowsHere + dIdx
        running[rowsHere] += 1
    del running, cum
    return sparse.csr_matrix((data, indices, indptr),
                             shape=(matrix_size, matrix_size),
                             dtype=np.float32)


def rebuildMatrix(pArrayOfTriangles, pWindowSize, pFlankingSize=None, pMaxDist=None, pStepsize=1, pSaveMemory=False, pWindowStarts=None, pMatrixSize=None, pProgress=True):
    """Fold predicted upper triangles back into one chromosome matrix.

    `pWindowStarts` is the sliding-window position each triangle came from. It
    defaults to 0, 1, 2, ... which is right whenever every window position was
    predicted. It is NOT right when the prediction was restricted to a subset of
    positions (`--includeRegions`): triangle i then belongs at position
    pWindowStarts[i], and assuming i would smear every prediction to the start
    of the chromosome. `pMatrixSize` must be given with it, because the number
    of triangles no longer determines the chromosome length.
    """
    log.info("Rebuilding matrix from triangles with window size %d", pWindowSize)
    # [Orijinal Kod - Korundu & Güncellendi]:
    # flankingSize = pFlankingSize if pFlankingSize is not None else pWindowSize
    # 2D TSS NPZ matris girdisinde varsayılan flanking boyutu 0'dır.
    flankingSize = pFlankingSize if pFlankingSize is not None else 0
    nr_matrices = pArrayOfTriangles.shape[0]
    if pWindowStarts is None:
        windowStarts = np.arange(nr_matrices, dtype=np.int64)
    else:
        windowStarts = np.asarray(pWindowStarts, dtype=np.int64)
        if windowStarts.shape[0] != nr_matrices:
            raise ValueError(
                "pWindowStarts has {:d} entries but there are {:d} triangles".format(
                    windowStarts.shape[0], nr_matrices))
    if pMatrixSize is not None:
        matrix_size = int(pMatrixSize)
    elif nr_matrices:
        matrix_size = int(windowStarts.max()) + (pWindowSize + 2 * flankingSize)
    else:
        matrix_size = pWindowSize + 2 * flankingSize

    log.info(f"Number of triangles: {nr_matrices}, window size: {pWindowSize}, flanking size: {flankingSize}")

    # Determine step size
    if pMaxDist is None or pMaxDist == pWindowSize:
        stepsize = 1
    else:
        stepsize = max(pStepsize, 1)
        stepsize = min(stepsize, pWindowSize - pMaxDist + 1)

    log.info(f"Stepsize set to {stepsize}")

    
    if pSaveMemory:
        # Band accumulator.
        #
        # The rebuilt matrix is banded by construction: a window of pWindowSize
        # bins can never place a value further than pWindowSize - 1 off the
        # diagonal. So the whole result fits in a dense (pWindowSize,
        # matrix_size) float32 array indexed by [offset-from-diagonal, row],
        # which for a human chromosome at 2 kb is a few hundred MB.
        #
        # The previous implementation accumulated COO blocks and folded each one
        # into a running CSR pair with `pSum = pSum + block_sum`. That is
        # quadratic in the number of blocks: the running matrix grows toward the
        # full band, and every flush reallocated and re-merged tens of millions
        # of entries that were already final. A chromosome took minutes, almost
        # all of it in those two additions.
        #
        # Here each window is a single in-place scatter-add into a fixed-size
        # buffer. Nothing grows, nothing is merged, and the sparse matrix is
        # built exactly once at the end. The per-window indices are the same
        # every time up to the offset j * pWindowSize, so they are computed once.
        #
        # The band is stored offset-major, [d, row] rather than [row, d],
        # because that is the layout in which a window's contribution to one
        # diagonal is contiguous. Only the flat scatter is used below, but the
        # layout is what makes the final CSR assembly a single pass.
        if pMaxDist is None or pMaxDist == pWindowSize:
            r, c = np.triu_indices(pWindowSize)
        else:
            r, c = np.mask_indices(pWindowSize, maskFunc, pMaxDist)
        r = r.astype(np.int64)
        c = c.astype(np.int64)
        entriesPerWindow = max(1, r.size)
        W = int(pWindowSize)

        # flat index into a (W, matrix_size) band: (c - r) picks the diagonal,
        # r picks the position within it. Adding j * 1 shifts along the
        # chromosome, so a window at j scatters to flatBase + j.
        flatBase = (c - r) * matrix_size + r
        log.info("Rebuilding %d windows into a %d x %d band (%d entries/window)",
                 nr_matrices, W, matrix_size, entriesPerWindow)

        # Only the SUM is accumulated. The divisor never has to be stored:
        # how many windows cover a cell depends on the window starts alone, and
        # for a cell on diagonal d at row p it is the number of starts in
        # [p - (W-1-d), p], which one prefix sum over the start positions
        # answers for every cell at once. That removes an array the size of the
        # band, and halves the work in the scatter loop.
        band_sum = np.zeros(W * matrix_size, dtype=np.float32)
        startCount = np.zeros(matrix_size, dtype=np.int32)

        for i in tqdm(range(0, nr_matrices, stepsize), desc="rebuilding matrix", disable=not pProgress):
            j = int(windowStarts[i]) + flankingSize
            tri = pArrayOfTriangles[i]
            vals = tri if tri.ndim == 1 else tri[r, c]
            # every (row, offset) pair within one window is distinct, so a plain
            # in-place add is correct here; np.add.at is not needed and is far
            # slower
            band_sum[flatBase + j] += vals
            startCount[j] += 1

        mean_matrix = bandToCsr(band_sum, startCount, W, matrix_size)
        del band_sum, startCount
    else:
        rows_all, cols_all, vals_all = [], [], []
        rows_count, cols_count, vals_count = [], [], []

        for i in tqdm(range(0, nr_matrices, stepsize), desc="rebuilding matrix", disable=not pProgress):
            j = int(windowStarts[i]) + flankingSize
            k = j + pWindowSize

            if pMaxDist is None or pMaxDist == pWindowSize:
                r, c = np.triu_indices(pWindowSize)
            else:
                r, c = np.mask_indices(pWindowSize, maskFunc, pMaxDist)

            rows = r + j
            cols = c + j

            tri = pArrayOfTriangles[i]

            # --- FIX: handle 1D or 2D triangles ---
            if tri.ndim == 1:
                vals = tri
            elif tri.ndim == 2:
                vals = tri[r, c]
            else:
                raise ValueError(f"Unexpected triangle shape: {tri.shape}")

            rows_all.append(rows)
            cols_all.append(cols)
            vals_all.append(vals)
            log.debug(f"Processed triangle {i}: added {len(vals)} values")

            rows_count.append(rows)
            cols_count.append(cols)
            vals_count.append(np.ones_like(vals, dtype=np.int32))
            log.debug(f"Count matrix updated for triangle {np.ones_like(vals, dtype=np.int32)}")
            # break
        # Combine all updates in a single sparse build
        rows_all = np.concatenate(rows_all)
        cols_all = np.concatenate(cols_all)
        vals_all = np.concatenate(vals_all, dtype=np.float32)
        rows_count = np.concatenate(rows_count)
        cols_count = np.concatenate(cols_count)
        vals_count = np.concatenate(vals_count)

        # Construct sparse sum and count matrices
        sum_matrix = sparse.coo_matrix(
            (vals_all, (rows_all, cols_all)), shape=(matrix_size, matrix_size), dtype=np.float32
        ).tocsr()
        count_matrix = sparse.coo_matrix(
            (vals_count, (rows_count, cols_count)), shape=(matrix_size, matrix_size), dtype=np.int32
        ).tocsr()

        # Convert once at the end
        # sum_matrix = sum_matrix.tocsr()
        # count_matrix = count_matrix.tocsr()

        mean_matrix = sum_matrix.copy().astype(np.float32)
        nonzero_mask = count_matrix.data != 0
        mean_matrix.data[nonzero_mask] = sum_matrix.data[nonzero_mask] / count_matrix.data[nonzero_mask]

    # return hic_diagonal_sharpen(mean_matrix)
    return mean_matrix  


def unsharp_mask_hic(matrix, sigma=1.0, strength=1.0):
    """
    Unsharp masking - good for TAD boundaries
    """
    if issparse(matrix):
        matrix = matrix.toarray()
    
    blurred = ndimage.gaussian_filter(matrix, sigma=sigma)
    sharpened = matrix + strength * (matrix - blurred)
    
    return np.clip(sharpened, 0, None)

def sharpen_hic(matrix, strength=0.5):
    """
    Laplacian sharpening for dense 2D array
    """
    if issparse(matrix):
        matrix = matrix.toarray()
    
    # Laplacian kernel
    laplacian = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])
    
    edges = ndimage.convolve(matrix, laplacian)
    sharpened = matrix + strength * edges
    
    # Ensure non-negative (Hi-C counts)
    sharpened = np.clip(sharpened, 0, None)
    
    return sharpened

def hic_diagonal_sharpen(matrix, strength=0.5, decay=0.3):
    """
    Stronger sharpening near diagonal (TADs), gentler for long-range
    """
    if issparse(matrix):
        matrix = matrix.toarray()
    
    size = matrix.shape[0]
    
    # Distance-from-diagonal weight matrix
    i, j = np.indices((size, size))
    distance = np.abs(i - j)
    weight = np.exp(-distance / (size * decay))
    
    # Laplacian edges
    laplacian = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])
    edges = ndimage.convolve(matrix, laplacian)
    
    # Weighted sharpening
    sharpened = matrix + strength * weight * edges
    
    return np.clip(sharpened, 0, None)


def rebuildChromosomeToFile(pJob):
    """Rebuild one chromosome from its predicted triangles and save it.

    Everything a chromosome needs is in `pJob`, and nothing is returned but the
    path, so this is safe to hand to a process pool: the 5 GB of triangles are
    read from disk by the worker (memory-mapped, so only the windows actually
    touched are paged in) and the assembled matrix is written by the worker.
    Neither ever crosses the process boundary.

    It lives in this module, and not in hicgan.predict, on purpose. A spawned
    worker re-imports the module that defines its target, and importing
    hicgan.predict would drag TensorFlow into every worker: seconds of start-up
    and hundreds of MB of RAM each, for code that only uses numpy and scipy.
    """
    (predictionFile, matrixFile, windowSize, flankingSize, saveMemory,
     windowStarts, matrixSize, targetValueRange, multiplier) = pJob
    predTriangles = np.load(predictionFile, mmap_mode="r")
    predMatrix = rebuildMatrix(pArrayOfTriangles=predTriangles,
                               pWindowSize=windowSize,
                               pFlankingSize=flankingSize,
                               pSaveMemory=saveMemory,
                               pWindowStarts=windowStarts,
                               pMatrixSize=matrixSize,
                               pProgress=False)
    del predTriangles
    if targetValueRange is not None:
        predMatrix = fromUnitRange(predMatrix, *targetValueRange)
    else:
        predMatrix = scaleArray(predMatrix) * multiplier
    if not sparse.isspmatrix(predMatrix):
        predMatrix = sparse.csr_matrix(predMatrix)
    sparse.save_npz(matrixFile, predMatrix.tocsr())
    nnz = int(predMatrix.nnz)
    del predMatrix
    return matrixFile, nnz


def rebuildChromosomeFromBand(pJob):
    """Turn one chromosome's accumulated band into a saved matrix chunk.

    The counterpart of rebuildChromosomeToFile for predictions that were summed
    on the GPU. Nothing is folded here: the summing already happened, so this
    only divides by the coverage and writes the sparse matrix.
    """
    (bandFile, matrixFile, windowSize, matrixSize, windowStarts,
     targetValueRange, multiplier) = pJob
    band = np.load(bandFile, mmap_mode="r")
    startCount = np.bincount(np.asarray(windowStarts, dtype=np.int64),
                             minlength=int(matrixSize))[:int(matrixSize)]
    predMatrix = bandToCsr(band, startCount, windowSize, matrixSize)
    del band
    if targetValueRange is not None:
        predMatrix = fromUnitRange(predMatrix, *targetValueRange)
    else:
        predMatrix = scaleArray(predMatrix) * multiplier
    if not sparse.isspmatrix(predMatrix):
        predMatrix = sparse.csr_matrix(predMatrix)
    sparse.save_npz(matrixFile, predMatrix.tocsr())
    nnz = int(predMatrix.nnz)
    del predMatrix
    return matrixFile, nnz


def rebuildProcessCount(pRequested, pWindowSize, pMatrixSizes, pJobCount):
    """How many chromosomes to rebuild at once without running out of RAM.

    A worker's peak is dominated by the dense band it accumulates into,
    windowSize x matrixSize float32, plus the CSR it builds from it. Two and a
    half times the band is a safe allowance. The largest chromosome in the job
    list sets the figure, because the pool may well be running it.

    pRequested of 0 means "decide from the machine". Anything else is taken as
    given, which is what a queueing system with its own memory limit wants.
    """
    if pRequested and int(pRequested) > 0:
        return max(1, min(int(pRequested), pJobCount))
    sizes = [s for s in (pMatrixSizes or []) if s]
    largest = max(sizes) if sizes else 0
    perWorker = 2.5 * int(pWindowSize) * largest * 4 if largest else 0
    available = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    byMemory = pJobCount
    if available and perWorker:
        # leave a third of what is free alone; this runs alongside other things
        byMemory = max(1, int((available * 0.66) // perWorker))
    byCpu = max(1, (os.cpu_count() or 2) - 1)
    n = max(1, min(pJobCount, byCpu, byMemory, 8))
    if perWorker:
        log.info("rebuilding with %d processes (~%.1f GB each, %.1f GB available)",
                 n, perWorker / 2**30, (available or 0) / 2**30)
    return n


def writeCooler(pMatrixList, pBinSizeInt, pOutfile, pChromosomeList, pChromSizeList=None,  pMetadata=None):
    #takes a matrix as numpy array or sparse matrix and writes a cooler matrix from it
    #modified from study project such that multiple chroms can be written to a single matrix

    # def pixelGenerator(pMatrixList, pOffsetList):
    #     '''
    #     yields pixel dataframes per Matrix
    #     Parameters:
    #     pMatrixList: list of matrices as np.ndarray or sparse.csr_matrix
    #     pOffsetList: list of integers that specify the offset into the bins dataframe

    #     Yields:
    #     pixels: pixels dataframe for all Hi-C matrices in the input list
    #     '''
    #     for matrix, offset in zip(pMatrixList, pOffsetList):
    #         log.info(f"Generating pixels for matrix with offset {offset}")
    #         #create the pixels for cooler
    #         triu_Indices = np.triu_indices(matrix.shape[0])
    #         pixels_tmp = pd.DataFrame(columns=['bin1_id','bin2_id','count'])
    #         pixels_tmp['bin1_id'] = (triu_Indices[0] + offset).astype("uint32")
    #         pixels_tmp['bin2_id'] = (triu_Indices[1] + offset).astype("uint32")
    #         readCounts = matrix[triu_Indices]
    #         if sparse.isspmatrix_csr(matrix): #for sparse matrices, slicing is different
    #             readCounts = np.transpose(readCounts)
    #         pixels_tmp['count'] = np.float64(readCounts)
    #         pixels_tmp.sort_values(by=['bin1_id','bin2_id'],inplace=True)
    #         yield pixels_tmp

    # def pixelGenerator(pMatrixList, pOffsetList):
    #     '''
    #     Efficiently yields pixel dataframes per Matrix, only for nonzero entries.
    #     '''
    #     for matrix, offset in zip(pMatrixList, pOffsetList):
    #         log.info(f"Generating pixels for matrix with offset {offset}")
    #         if sparse.isspmatrix(matrix):
    #             # Use COO format for easy access to nonzero indices
    #             coo = matrix.tocoo()
    #             mask = coo.row <= coo.col  # upper triangle
    #             bin1 = coo.row[mask] + offset
    #             bin2 = coo.col[mask] + offset
    #             count = coo.data[mask]
    #         else:
    #             # Dense matrix: get upper triangle indices
    #             triu = np.triu_indices(matrix.shape[0])
    #             bin1 = triu[0] + offset
    #             bin2 = triu[1] + offset
    #             count = matrix[triu]
    #             mask = count != 0
    #             bin1 = bin1[mask]
    #             bin2 = bin2[mask]
    #             count = count[mask]
    #         pixels_tmp = pd.DataFrame({
    #             'bin1_id': bin1.astype("uint32"),
    #             'bin2_id': bin2.astype("uint32"),
    #             'count': np.float64(count)
    #         })
    #         pixels_tmp.sort_values(by=['bin1_id','bin2_id'], inplace=True)
    #         yield pixels_tmp
    def _load_matrix_entry(matrix_entry):
        if isinstance(matrix_entry, str):
            if matrix_entry.endswith(".npz"):
                return sparse.load_npz(matrix_entry)
            if matrix_entry.endswith(".npy"):
                return np.load(matrix_entry)
            raise ValueError(f"Unsupported matrix file format: {matrix_entry}")
        return matrix_entry

    def pixelGenerator(pMatrixList, pOffsetList, sort_pixels=False, dtype=np.float32):
        """
        Memory-efficient generator that yields pixel DataFrames from sparse Hi-C matrices.
        Only upper-triangular nonzero entries are returned.
        
        Parameters
        ----------
        pMatrixList : list of scipy.sparse matrices or dense arrays
            List of contact matrices. Dense inputs are converted to sparse on the fly.
        pOffsetList : list of int
            Genomic bin offsets corresponding to each matrix.
        sort_pixels : bool, optional
            Whether to sort each pixel DataFrame by (bin1_id, bin2_id). Default False.
        dtype : np.dtype, optional
            Floating-point precision for counts (default float32 for memory efficiency).
        
        Yields
        ------
        pandas.DataFrame
            Columns: ['bin1_id', 'bin2_id', 'count']
        """
        for matrix_entry, offset in zip(pMatrixList, pOffsetList):
            matrix = _load_matrix_entry(matrix_entry)
            log.debug(f"Generating pixels for matrix with offset {offset}")

            # Ensure matrix is sparse in COO format
            if not sparse.isspmatrix(matrix):
                log.debug("Converting dense matrix to sparse COO")
                matrix = sparse.coo_matrix(matrix)
            else:
                matrix = matrix.tocoo(copy=False)

            # Keep only upper triangle (including diagonal)
            mask = matrix.row <= matrix.col
            if not np.any(mask):
                continue
            log.debug(f"Number of nonzero upper-triangular entries: {np.sum(mask)}")
            bin1 = matrix.row[mask].astype(np.uint32, copy=False) + offset
            bin2 = matrix.col[mask].astype(np.uint32, copy=False) + offset
            count = matrix.data[mask].astype(dtype, copy=False)

            # Create DataFrame without extra copies
            pixels_tmp = pd.DataFrame({
                'bin1_id': bin1,
                'bin2_id': bin2,
                'count': count
            })

            if sort_pixels and len(pixels_tmp) > 1:
                pixels_tmp.sort_values(['bin1_id', 'bin2_id'], inplace=True, ignore_index=True)

            yield pixels_tmp
            del matrix

        # log.info("Sorting pixel DataFrame if requested")
        # if sort_pixels and len(pixels_tmp) > 1:
        #     pixels_tmp.sort_values(['bin1_id', 'bin2_id'], inplace=True, ignore_index=True)

        # yield pixels_tmp

    if pMatrixList is None or pChromosomeList is None or pBinSizeInt is None or pOutfile is None:
        msg = "input empty. No cooler matrix written"
        log.error(msg)
        print(msg)
        return
    if len(pMatrixList) != len(pChromosomeList):
        msg = "number of input arrays and chromosomes must be the same"
        log.error(msg)
        print(msg)
        return
    if pChromSizeList is not None and len(pChromSizeList) != len(pChromosomeList):
        msg = "if chrom sizes are given, they must be provided for ALL chromosomes"
        log.error(msg)
        print(msg)
        return
    bins = pd.DataFrame(columns=['chrom','start','end'])
    offsetList = [0]
    for i, (matrix_entry, chrom) in enumerate(zip(pMatrixList,pChromosomeList)):
        matrix = _load_matrix_entry(matrix_entry)
        log.info(f"Processing chromosome {chrom} ({i+1}/{len(pChromosomeList)})")
        #the chromosome size may not be integer-divisible by the bin size
        #so specifying the real chrom size is possible, but the
        #number of bins must still correspond to the matrix size
        chromSizeInt = int(matrix.shape[0] * pBinSizeInt)
        if pChromSizeList is not None \
                and pChromSizeList[i] is not None \
                and pChromSizeList[i] > (chromSizeInt - pBinSizeInt)\
                and pChromSizeList[i] < chromSizeInt:
            chromSizeInt = int(pChromSizeList[0])

        #create the bins for cooler
        bins_tmp = pd.DataFrame(columns=['chrom','start','end'])
        binStartList = list(range(0, chromSizeInt, int(pBinSizeInt)))
        binEndList = list(range(int(pBinSizeInt), chromSizeInt, int(pBinSizeInt)))
        binEndList.append(chromSizeInt)
        bins_tmp['start'] = np.uint32(binStartList)
        bins_tmp['end'] = np.uint32(binEndList)
        bins_tmp["chrom"] = str(chrom)
        bins = pd.concat([bins, bins_tmp], ignore_index=True)
        # bins = bins.append(bins_tmp, ignore_index=True)
        offsetList.append(offsetList[-1] + bins_tmp.shape[0])
        del matrix

    #correct dtypes for joint dataframe
    bins["start"] = bins["start"].astype("uint32")
    bins["end"] = bins["end"].astype("uint32")
    offsetList = offsetList[:-1] #don't need the last one, no more matrix to follow

    log.info(f"Writing cooler file to {pOutfile}")

    #write out the cooler
    cooler.create_cooler(pOutfile, bins=bins, pixels=pixelGenerator(pMatrixList=pMatrixList, pOffsetList=offsetList), dtypes={'count': np.float64}, ordered=True, metadata=pMetadata)

def distanceNormalize(pSparseCsrMatrix, pWindowSize_bins):
    #compute the means along the diagonals (= same distance)
    #and divide all values on the diagonals by their respective mean
    diagList = []
    for i in range(pWindowSize_bins):
        diagArr = sparse.csr_matrix.diagonal(pSparseCsrMatrix,i)
        diagList.append(diagArr/diagArr.mean())
    distNormalizedMatrix = sparse.diags(diagList,np.arange(pWindowSize_bins),format="csr")
    return distNormalizedMatrix

def plotChromatinFactors(pFactorArray, pFeatureNameList, 
                            pChromatinFolder, pChrom, pBinsize, pStartbin,
                            pOutputPath, pPlotType, pFigureType="png"):
    #plot box- or line plots of the chromatin factors stored in pFactorDict
    #the matrices are required to determine the binsize for the line plots
    if pPlotType == "box":
        plotFn = plotChromatinFactors_boxplots
    elif pPlotType == "line":
        plotFn = plotChromatinFactors_lineplots
    else:
        return
 
    filename = "chromFactors_{:s}_{:s}_{:s}.{:s}".format(pPlotType, pChromatinFolder.rstrip("/").replace("/","-"), str(pChrom), pFigureType)
    filename = os.path.join(pOutputPath,filename)
    plotTitle = "Chromosome {:s} | Dir. {:s}".format(str(pChrom),pChromatinFolder)
    plotFn(pChromFactorArray=pFactorArray,
                    pFilename=filename, 
                    pBinSize=pBinsize,
                    pStartbin=pStartbin,
                    pAxTitle=plotTitle, 
                    pFactorNames=pFeatureNameList)

def plotChromatinFactors_boxplots(pChromFactorArray, pFilename, pBinSize=None, pStartbin=None, pAxTitle=None, pFactorNames=None):
    #store box plots of the chromatin factors in the array
    fig1, ax1 = plt.subplots()
    toPlotList = []
    for i in range(pChromFactorArray.shape[1]):
        toPlotList.append(pChromFactorArray[:,i])
    ax1.boxplot(toPlotList)
    fig1.suptitle("Chromatin factor boxplots")
    if pAxTitle is not None:
        ax1.set_title(str(pAxTitle))
    if pFactorNames is not None \
            and isinstance(pFactorNames,list) \
            and len(pFactorNames) == pChromFactorArray.shape[1]:
        ax1.set_xticklabels(pFactorNames, rotation=90)
    ax1.set_xlabel("Chromatin factor")
    ax1.set_ylabel("Chromatin factor signal value")
    fig1.tight_layout()
    fig1.savefig(pFilename)
    plt.close(fig1)
    del fig1, ax1

def plotChromatinFactors_lineplots(pChromFactorArray, pFilename, pBinSize, pStartbin, pAxTitle=None, pFactorNames=None):
    #plot chromatin factors line plots
    #for debugging purposes only, not for production use
    winsize = pChromFactorArray.shape[0]
    nr_subplots = pChromFactorArray.shape[1]
    x_axis_values = np.arange(winsize) * pBinSize
    figsizeX = max(30, int(max(x_axis_values)/2000000))
    figsizeX = min(100, figsizeX)
    figsizeY = max(6, 3*nr_subplots)
    figsizeY = min(100, figsizeY)
    if isinstance(pStartbin, int):
        x_axis_values += pStartbin * pBinSize
    fig1, axs1 = plt.subplots(nr_subplots, 1, sharex = True, figsize=(figsizeX, figsizeY))
    for i in range(nr_subplots):
        axs1[i].plot(x_axis_values, pChromFactorArray[:,i])
        axs1[i].grid(True)
        #try to plot a reasonable number of major x-axis ticks
        if max(x_axis_values) < 1000000:
            locVal = 50000
        elif max(x_axis_values) < 10000000:
            locVal = 500000
        elif max(x_axis_values) < 50000000:
            locVal = 2500000
        elif max(x_axis_values) < 100000000:
            locVal = 5000000
        else:
            locVal = 10000000
        axs1[i].xaxis.set_major_locator(MultipleLocator(locVal))
        if pFactorNames is not None \
                and isinstance(pFactorNames,list) \
                and len(pFactorNames) == nr_subplots:
            axs1[i].set_xlabel(pFactorNames[i])
    if pAxTitle is not None:
        fig1.text(0.5, 0.04, str(pAxTitle), ha='center')
    axs1[0].set_xlim([min(x_axis_values), max(x_axis_values)])
    fig1.tight_layout()
    fig1.text(0.04, 0.5, 'signal value', va='center', rotation='vertical')
    fig1.suptitle("Chromatin factors")
    fig1.savefig(pFilename)
    plt.close(fig1)
    del fig1, axs1

def clampArray(pArray):
    #clamp all values in pArray to be within 
    #lowerQuartile - 1.5xInterquartile ... upperQuartile + 1.5xInterquartile
    clampedArray = pArray.copy()
    upperQuartile = np.quantile(pArray,0.75)
    lowerQuartile = np.quantile(pArray,0.25)
    interQuartile = upperQuartile - lowerQuartile
    if interQuartile > 1.0:
        upperClampingBound = upperQuartile + 1.5*interQuartile
        lowerClampingBound = lowerQuartile - 1.5*interQuartile
        clampedArray[clampedArray < lowerClampingBound] = lowerClampingBound
        clampedArray[clampedArray > upperClampingBound] = upperClampingBound
    return clampedArray
def computePearsonCorrelation(pCoolerFile1, pCoolerFile2, 
                              pWindowsize_bp,
                              pModelChromList, pTargetChromStr,
                              pModelCellLineList, pTargetCellLineStr,
                              pPlotOutputFile=None, pCsvOutputFile=None):
    '''
    compute distance-stratified pearson correlation for target chromosome
    directly from cooler files and plot or write to file

    Parameters:
        pCoolerFile1 (str): Path to cooler file 1
        pCoolerFile2 (str): Path to cooler file 2
        pWindowsize_bp (int): Windowsize in basepairs for which correlations shall be computed
        pModelChromList (list): List of strings, will appear in plot title
        pModelCellLineList (list): List of strings, will appear in plot title
        pTargetChromStr (str): the target chromosome, e.g. >chr10< or >10<
        pTargetCellLineStr (str): the target cell line, will appear in plot title
        pPlotOutputFile (str): filename of correlation plot
        pCsvOutputFile (str): filename of correlation csv file
    
    Returns:
        None
    ''' 

    sparseMatrix1, binsize1 = getMatrixFromCooler(pCoolerFile1, pTargetChromStr)
    sparseMatrix2, binsize2 = getMatrixFromCooler(pCoolerFile2, pTargetChromStr)
    errorMsg = ""
    if sparseMatrix1 is None:
        errorMsg += "Chrom {:s} could not be loaded from {:s}\n"
        errorMsg = errorMsg.format(str(pTargetChromStr), pCoolerFile1)
    if sparseMatrix2 is None:
        errorMsg += "Chrom {:s} could not be loaded from {:s}\n"
        errorMsg = errorMsg.format(str(pTargetChromStr), pCoolerFile2)
    if errorMsg != "":
        errorMsg += "Potential reasons: Wrong file format, wrong chromosome naming scheme or chromosome missing"
        raise SystemExit(errorMsg)
    if binsize1 != binsize2:
        errorMsg = "Aborting. Binsizes of matrices are not equal\n"
        errorMsg += "{:s} -- {:d}bp\n"
        errorMsg += "{:s} -- {:d}bp\n"
        errorMsg = errorMsg.format(pCoolerFile1,binsize1, pCoolerFile2, binsize2)
        raise SystemExit(errorMsg)
    resultsDf = computePearsonCorrelationSparse(pSparseCsrMatrix1= sparseMatrix1,
                                                pSparseCsrMatrix2= sparseMatrix2,
                                                pBinsize= binsize1,
                                                pWindowsize_bp= pWindowsize_bp,
                                                pModelChromList= pModelChromList,
                                                pTargetChromStr= pTargetChromStr,
                                                pModelCellLineList= pModelCellLineList,
                                                pTargetCellLineStr= pTargetCellLineStr)
    if pCsvOutputFile is not None:
        resultsDf.to_csv(pCsvOutputFile)
    if pPlotOutputFile is not None:
        plotPearsonCorrelationDf(pResultsDfList=[resultsDf], 
                                 pLegendList=["Pearson corr."],
                                 pOutfile=pPlotOutputFile,
                                 pMethod="pearson")
    return resultsDf

def computePearsonCorrelationSparse(pSparseCsrMatrix1, pSparseCsrMatrix2, 
                                    pBinsize, pWindowsize_bp, 
                                    pModelChromList, pTargetChromStr, 
                                    pModelCellLineList, pTargetCellLineStr):
    '''
    compute distance-stratified Pearson correlation from two sparse matrices

    Parameters:
        pSparseCsrMatrix1 (scipy.sparse.csr_matrix): sparse csr matrix 1
        pSparseCsrMatrix2 (scipy.sparse.csr_matrix): sparse csr matrix 2
        pBinsize (int): the binsize of each bin in the sparse matrices
        pWindowsize_bp (int): the windowsize in basepairs for which correlations shall be computed
        pModelChromList (list): list of strings, will appear in plot title
        pTargetChromStr (str): the target chromosome, e.g. >chr10< or >10<
        pTargetCellLineStr (str): the target cell line, will appear in plot title
        pModelCellLineList (list): List of strings, will appear in plot title

    Returns:
        (pandas.DataFrame): Pandas dataframe containing the correlations per distance 
    '''
    numberOfDiagonals = int(np.round(pWindowsize_bp/pBinsize))
    if numberOfDiagonals < 1:
        msg = "Window size must be larger than bin size of matrices.\n"
        msg += "Remember to specify window in basepairs, not bins."
        raise SystemExit(msg)
    shape1 = pSparseCsrMatrix1.shape
    shape2 = pSparseCsrMatrix2.shape
    if shape1 != shape2:
        msg = "Aborting. Shapes of matrices are not equal.\n"
        msg += "Shape 1: ({:d},{:d}); Shape 2: ({:d},{:d})"
        msg = msg.format(shape1[0],shape1[1],shape2[0],shape2[1])
        raise SystemExit(msg)
    if numberOfDiagonals > shape1[0]-1:
        msg = "Aborting. Window size {0:d} larger than matrix size {:d}"
        msg = msg.format(numberOfDiagonals, shape1[0]-1)
        raise SystemExit(msg)
    
    trapezIndices = np.mask_indices(shape1[0],maskFunc,k=numberOfDiagonals)
    reads1 = np.array(pSparseCsrMatrix1[trapezIndices])[0]
    reads2 = np.array(pSparseCsrMatrix2[trapezIndices])[0]

    matrixDf = pd.DataFrame(columns=['first','second','distance','reads1','reads2'])
    matrixDf['first'] = np.uint32(trapezIndices[0])
    matrixDf['second'] = np.uint32(trapezIndices[1])
    matrixDf['distance'] = np.uint32(matrixDf['second'] - matrixDf['first'])
    matrixDf['reads1'] = np.float32(reads1)
    matrixDf['reads2'] = np.float32(reads2)
    matrixDf.fillna(0, inplace=True)

    pearsonAucIndices, pearsonAucValues = getCorrelation(matrixDf,'distance', 'reads1', 'reads2', 'pearson')
    pearsonAucScore = metrics.auc(pearsonAucIndices, pearsonAucValues)
    spearmanAucIncides, spearmanAucValues = getCorrelation(matrixDf,'distance', 'reads1', 'reads2', 'spearman')
    spearmanAucScore = metrics.auc(spearmanAucIncides, spearmanAucValues)
    # print("PearsonAUC: {:.3f}".format(pearsonAucScore))
    # print("SpearmanAUC: {:.3f}".format(spearmanAucScore))

    columns = ["corrMeth", "modelChroms", "targetChrom", 
                           "modelCellLines", "targetCellLine", 
                           "R2", "MSE", "MAE", "MSLE", "AUC",
                           "binsize", "windowsize"]
    columns.extend(sorted(list(matrixDf.distance.unique())))
    resultsDf = pd.DataFrame(columns=columns)
    resultsDf["corrMeth"] = ["pearson", "spearman"]
    resultsDf.set_index("corrMeth", inplace=True)
    resultsDf.loc[:, 'modelChroms'] = ", ".join([str(x) for x in pModelChromList])
    resultsDf.loc[:, 'targetChrom'] = pTargetChromStr
    resultsDf.loc[:, 'modelCellLines'] = ", ".join([str(x) for x in pModelCellLineList])
    resultsDf.loc[:, 'targetCellLine'] = pTargetCellLineStr
    resultsDf.loc[:, "R2"] = metrics.r2_score(matrixDf['reads2'], matrixDf['reads1'])
    resultsDf.loc[:, 'MSE'] = metrics.mean_squared_error( matrixDf['reads2'], matrixDf['reads1'])
    resultsDf.loc[:, 'MAE'] = metrics.mean_absolute_error( matrixDf['reads2'], matrixDf['reads1'])
    resultsDf.loc[:, 'MSLE'] = metrics.mean_squared_log_error(matrixDf['reads2'], matrixDf['reads1'])
    resultsDf.loc['pearson', 'AUC'] = pearsonAucScore 
    resultsDf.loc['spearman', 'AUC'] = spearmanAucScore
    resultsDf.loc[:, 'binsize'] = pBinsize
    resultsDf.loc[:, 'windowsize'] = pWindowsize_bp
    
    for pearsonIndex, corrValue in zip(pearsonAucIndices,pearsonAucValues):
        columnName = int(round(pearsonIndex * matrixDf.distance.max()))
        resultsDf.loc["pearson", columnName] = corrValue
    for spearmanIndex, corrValue in zip(spearmanAucIncides,spearmanAucValues):
        columnName = int(round(spearmanIndex * matrixDf.distance.max()))
        resultsDf.loc["spearman", columnName] = corrValue
    return resultsDf
    
def plotPearsonCorrelationDf(pResultsDfList, pLegendList, pOutfile, pMethod="pearson"):
    #helper function to plot distance-stratified Pearson correlation stored in pandas dataframes
    if pMethod not in ["pearson", "spearman"]:
        print("plotting only supported for 'pearson' and 'spearman' correlation methods")
        return
    if pResultsDfList is None or pLegendList is None:
        return
    if not isinstance(pResultsDfList,list) or not isinstance(pLegendList,list):
        return
    legendStrList = [str(x) for x in pLegendList]
    if len(pResultsDfList) != len(legendStrList):
        msg = "can't plot, too many / too few legends\n"
        msg += "no. of legend entries should be: {:d}, given {:d}"
        msg = msg.format(len(pResultsDfList), len(legendStrList))
        print(msg)
        return
    
    fig1, ax1 = plt.subplots()
    ax1.set_ylabel("{:s} correlation".format(pMethod[0].upper() + pMethod[1:] ))
    ax1.set_xlabel("Genomic distance / Mbp")
    trainChromSet = set()
    targetChromSet = set()
    trainCellLineSet = set()
    targetCellLineSet = set()
    maxXVal = 0
    for i, resultsDf in enumerate(pResultsDfList):
        try:
            resolutionInt = int(resultsDf.loc[pMethod, 'binsize'])
            windowsize_bp = int(resultsDf.loc[pMethod, 'windowsize'])
            trainChromSet.add(resultsDf.loc[pMethod, 'modelChroms'])
            targetChromSet.add(resultsDf.loc[pMethod, 'targetChrom'])
            trainCellLineSet.add(resultsDf.loc[pMethod, 'modelCellLines'])
            targetCellLineSet.add(resultsDf.loc[pMethod, 'targetCellLine'])
            area_under_corr_curve = resultsDf.loc[pMethod, 'AUC']
            maxDist_bp = int(windowsize_bp / resolutionInt)
            columnNameList = [x for x in range(maxDist_bp)]
            corrXValues = np.arange(maxDist_bp) * resolutionInt / 1000000
            corrYValues = resultsDf.loc[pMethod, columnNameList].values.astype("float32")
        except Exception as e:
            msg = str(e) + "\n"
            msg += "results dataframe {:d} does not contain all relevant fields (binsize, distance stratified pearson correlation data etc.)"
            msg = msg.format(i)
            print(msg)
        label = pLegendList[i]
        if label is None:
            label = pMethod + " / AUC: {:.3f}".format(area_under_corr_curve)
        else:
            label = label + " / AUC: {:.3f}".format(area_under_corr_curve)
        ax1.plot(corrXValues, corrYValues, label = label)
        maxXVal = max(maxXVal, corrXValues[-1])
    titleStr = "Pearson correlation vs. genomic distance"
    if len(trainChromSet) == len(targetChromSet) == len(trainCellLineSet) == len(targetCellLineSet) == 1:
        titleStr += "\n {:s}, {:s} on {:s}, {:s}"
        titleStr = titleStr.format(list(trainCellLineSet)[0], list(trainChromSet)[0], list(targetCellLineSet)[0], list(targetChromSet)[0])
    ax1.set_title(titleStr)
    ax1.set_ylim([0,1])
    ax1.set_xlim([0,maxXVal])
    ax1.grid(True)
    ax1.legend(frameon=False, loc="upper right")
    
    if pOutfile is None:
        outfile = "correlation.png"
        fig1.savefig(outfile)
    else:
        outfile = pOutfile
        if os.path.splitext(outfile)[1] not in ['.png', '.svg', '.pdf']:
            outfile = os.path.splitext(pOutfile)[0] + '.png'
            msg = "Outfile must have png, pdf or svg file extension.\n"
            msg += "Renamed outfile to {:s}".format(outfile)
            print(msg)
        fig1.savefig(outfile)
    plt.close(fig1)
    del fig1, ax1

def maskFunc(pArray, pWindowSize=0):
    #mask a trapezoid along the (main) diagonal of a 2D array
    #this code is copied from the study project by Ralf Krauth
    #https://github.com/MasterprojectRK/HiCPrediction/blob/master/hicprediction/createTrainingSet.py
    maskArray = np.zeros(pArray.shape)
    upperTriaInd = np.triu_indices(maskArray.shape[0]) # pylint: disable=unsubscriptable-object
    notRequiredTriaInd = np.triu_indices(maskArray.shape[0], k=pWindowSize) # pylint: disable=unsubscriptable-object
    maskArray[upperTriaInd] = 1
    maskArray[notRequiredTriaInd] = 0
    return maskArray

def getCorrelation(pData, pDistanceField, pTargetField, pPredictionField, pCorrMethod):
    """
    Helper method to calculate correlation
    This method has originally been written by Andre Bajorat during his study project,
    licensed under the MIT License: 
    https://github.com/abajorat/HiCPrediction/blob/master/hicprediction/predict.py
    It has been adapted by Ralf Krauth during his study project:
    https://github.com/MasterprojectRK/HiCPrediction/blob/master/hicprediction/predict.py
    
    Parameters:
        pData (pandas.DataFrame): Pandas dataframe with read counts / distances
        pDistanceField (str): the column name of the distance Field in the dataframe
        pTargetField (str): the column name of the target read counts in the dataframe
        pPredictionField (str): column name of the predicted read counts in the dataframe
        pCorrMethod (str): any of the correlation methods supported by pandas DataFrame corr method
    
    Returns:
        indices (list): integer list of index values 
        values (list): float list of correlation values
    """
    # print(pData)
    # pData.to_csv("data.csv")
    # new = pData.groupby(pDistanceField, group_keys=False)[[pTargetField,
    #     pPredictionField]].corr(method=pCorrMethod)
    
    # print(new)

    # new = new.iloc[0::2,-1]
    # new.reset_index(drop=True, inplace=True)
    # print(new)

    # #sometimes there is no variation in prediction / target per distance, then correlation is NaN
    # #need to drop these, otherwise AUC will be NaN, too.
    # new.dropna(inplace=True) 
    # values = new.values
    # print(new)
    # indices = new.index.tolist()
    # indices = list(map(lambda x: x[0], indices))
    # indices = np.array(indices, dtype=np.int32)
    # div = pData[pDistanceField].max()
    # indices = indices / div 
    # return indices, values
    correlation_matrix = pData.groupby(pDistanceField, group_keys=False)[[pTargetField, pPredictionField]].corr(method=pCorrMethod)

    # Step 2: Extracting Relevant Correlation Values
    correlation_values = correlation_matrix.iloc[0::2, -1].reset_index(drop=True)

    # Step 3: Handling NaNs
    correlation_values.dropna(inplace=True)

    # Step 4: Preparing the Output
    indices = correlation_values.index.tolist()
    indices = np.array(indices, dtype=np.int32)
    max_distance = pData[pDistanceField].max()
    indices = indices / max_distance

    values = correlation_values.values

    # Print statements for debugging
    # print(correlation_matrix)
    # print(correlation_values)
    # print(indices, values)

    # Return the final indices and values
    return indices, values

def getChromPrefixCooler(pCoolerFileName):
    '''
    check if the chromosomes in the cooler file 
    start with 'chr' or not; e.g. 'chr10' vs. '10'
    '''
    try:
        coolerMatrix = cooler.Cooler(pCoolerFileName) 
        chromSizes = coolerMatrix.chromsizes.to_dict()
        chromNameList = [entry for entry in chromSizes]
    except Exception as e:
        raise(e)
    prefix = None
    if chromNameList is not None and len(chromNameList) > 0 and str(chromNameList[0]).startswith("chr"):
        prefix = "chr"
    elif  chromNameList is not None and len(chromNameList) > 0:
        prefix = ""
    else:
        msg = "No valid entries found in cooler file {:s}"
        msg = msg.format(pCoolerFileName)
        raise ValueError(msg) 
    return prefix

def getDiamondIndices(pMatsize, pDiamondsize):
    nr_diamonds = pMatsize - 2*pDiamondsize
    if nr_diamonds <= 1:
        msg = "Diamondsize too large for Matsize"
        raise ValueError(msg)
    start_offset = pDiamondsize
    rowEndList = [i + start_offset for i in range(nr_diamonds)]
    rowStartList = [i-pDiamondsize for i in rowEndList] 
    columnStartList = [i+1 for i in rowEndList]
    columnEndList = [i+pDiamondsize for i in columnStartList]
    return rowStartList, rowEndList, columnStartList, columnEndList

def saveInsulationScoreToBedgraph(scoreArrayList, binsize, diamondsize, chromosomeList, filename, chromSizeList=None, startbinList=None):
    if not isinstance(scoreArrayList, list) \
            or not isinstance(chromosomeList, list):
        msg = "Warning: not saving insulation scores to bedgraph. Wrong input format"
        print(msg)
        return
    if len(scoreArrayList) != len(chromSizeList):
        msg = "Warning: not saving insulation scores to bedgraph. Inconsistent input lengths"
        print(msg)
        return
    if startbinList is not None and not isinstance(startbinList,list) \
            or (isinstance(startbinList, list) and len(startbinList) != len(scoreArrayList)):
        msg = "Warning: not saving insulation scores to bedgraph. Bad startbin list"
        print(msg)
        return
    if chromSizeList is not None and not isinstance(chromSizeList, list) \
            or (isinstance(chromSizeList, list) and len(chromSizeList) != len(scoreArrayList)):
        msg = "Warning: not saving insulation scores to bedgraph. Bad chromsize list"
        print(msg)
        return
    if not isinstance(binsize, int) or not isinstance(diamondsize, int):
        msg = "binsize and diamondsize must be int"
        print(msg)
        return
    if startbinList is None:
        startbinList = [0]*len(scoreArrayList)
    if chromSizeList is None:
        chromSizeList = [(score.shape[0] + 2*diamondsize)*binsize for score in scoreArrayList]
    dfList = []
    for chromSize, scoreArray, chromosome, startbin in zip(chromSizeList, scoreArrayList, chromosomeList, startbinList):
        posList = [i for i in range(0,chromSize,binsize)] + [chromSize]
        startList = [i for i, j in zip(posList, posList[1:])]
        endList = [j for i, j in zip(posList, posList[1:])]
        scores = [0]*diamondsize + list(scoreArray) + [0]*diamondsize
        if len(scores) != len(startList):
            msg = "Score Array wrong size"
            print(msg)
            continue
        df = pd.DataFrame(columns=["chrom", "chromStart", "chromEnd", "dataValue"])
        df["chromStart"] = startList
        df["chromEnd"] = endList
        df["dataValue"] = scores
        df["chrom"] = chromosome
        if isinstance(startbin, int):
            df["chromStart"] += (startbin * binsize)
            df["chromEnd"] += (startbin * binsize)
        dfList.append(df)
    df = pd.concat(dfList, ignore_index=True)
    with open(filename, "w") as bgf:
            bgf.write("track type=bedGraph\n")
            df.to_csv(bgf, sep="\t", header=False, index=False)

def computeScore(pMatrix, pDiamondsize):
    if not isinstance(pDiamondsize, int):
        msg = "Warning: Cannot compute score; size for score computation must be integer"
        print(msg)
        return
    if not isinstance(pMatrix, np.ndarray) or len(pMatrix.shape) != 2 or pMatrix.shape[0] - 2*pDiamondsize <= 1:
        msg = "Warning: Cannot compute score; matrix wrong format or bad input shape or score size too large"
        print(msg)
        return
    rowStartList, rowEndList, columnStartList, columnEndList = getDiamondIndices(pMatsize=pMatrix.shape[0], pDiamondsize=pDiamondsize)
    l = [ pMatrix[i:j,k:l] for i,j,k,l in zip(rowStartList,rowEndList,columnStartList,columnEndList) ]
    return np.array([ np.mean(i) for i in l ]).astype("float32")   