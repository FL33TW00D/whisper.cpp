import numpy as np
from numba import jit
from enum import Enum
from tqdm import tqdm

EXPORT_DIR = "."


class QuantizationScheme(Enum):
    Q4 = "q4"
    Q8 = "q8"
    PF16 = "pf16"  # 2xF16 -> U32

    def __str__(self):
        return self.value


@jit(nopython=True)
def q4(matrix):
    """
    Quantize a matrix of float32 values to sint4.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix to quantize.
    M : int
        The number of rows in the matrix.
    N : int
        The number of columns in the matrix.

    Returns
    -------
    numpy.ndarray
        The quantized matrix.
    """
    M = matrix.shape[0]
    N = matrix.shape[1]

    block_size = 8
    # [768, 768] -> [768, 96] + [768, 96]
    quantized_matrix = np.zeros((M, N // block_size), dtype=np.uint32)
    absmax_matrix = np.zeros((M, N // block_size), dtype=np.float32)

    # Quantize the matrix values to sint8 and pack them into uint32
    for i in range(M):
        for j in range(0, N, block_size):
            local_absmax = np.max(np.abs(matrix[i, j : j + block_size]))
            absmax_matrix[i, j // block_size] = local_absmax

            packed_value = (
                (round(matrix[i, j] / local_absmax * 7) & 0x0F)
                | ((round(matrix[i, j + 1] / local_absmax * 7) & 0x0F) << 4)
                | ((round(matrix[i, j + 2] / local_absmax * 7) & 0x0F) << 8)
                | ((round(matrix[i, j + 3] / local_absmax * 7) & 0x0F) << 12)
                | ((round(matrix[i, j + 4] / local_absmax * 7) & 0x0F) << 16)
                | ((round(matrix[i, j + 5] / local_absmax * 7) & 0x0F) << 20)
                | ((round(matrix[i, j + 6] / local_absmax * 7) & 0x0F) << 24)
                | ((round(matrix[i, j + 7] / local_absmax * 7) & 0x0F) << 28)
            )
            quantized_matrix[i, j // block_size] = packed_value

    return (quantized_matrix, absmax_matrix)


@jit(nopython=True)
def q8(matrix):
    """
    Quantize a matrix of float32 values to sint8.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix to quantize.
    M : int
        The number of rows in the matrix.
    N : int
        The number of columns in the matrix.

    Returns
    -------
    numpy.ndarray
        The quantized matrix.
    """
    M = matrix.shape[0]
    N = matrix.shape[1]

    block_size = 4
    # [768, 768] -> [768, 192]
    quantized_matrix = np.zeros((M, N // block_size), dtype=np.uint32)

    absmax = np.max(np.abs(matrix))

    # Quantize the matrix values to sint8 and pack them into uint32
    for i in range(M):
        for j in range(0, N, block_size):
            packed_value = (
                (round(matrix[i, j] / absmax * 127) & 0xFF)
                | ((round(matrix[i, j + 1] / absmax * 127) & 0xFF) << 8)
                | ((round(matrix[i, j + 2] / absmax * 127) & 0xFF) << 16)
                | ((round(matrix[i, j + 3] / absmax * 127) & 0xFF) << 24)
            )
            quantized_matrix[i, j // block_size] = packed_value

    return (quantized_matrix, np.array(absmax, dtype=np.float32))


def pf16(matrix):
    """
    Quantize a matrix of float32 values to pf16.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix to quantize.
    M : int
        The number of rows in the matrix.
    N : int
        The number of columns in the matrix.

    Returns
    -------
    numpy.ndarray (dtype=np.uint32)
    """
    M = matrix.shape[0]
    N = matrix.shape[1]

    # Ensure matrix has even number of columns.
    assert N % 2 == 0

    quantized_matrix = np.zeros((M, N // 2), dtype=np.uint32)
    matrix = matrix.astype(np.float16)

    for i in tqdm(range(M)):
        for j in range(0, N, 2):
            float0 = np.uint16(matrix[i, j]) 
            float1 = np.uint16(matrix[i, j + 1])

            # Pack two uint16 into a uint32
            packed_value = (float1 << 16) | float0 

            quantized_matrix[i, j // 2] = packed_value
    return quantized_matrix 


@jit(nopython=True)
def dequant_pf16(matrix):
    """
    Dequantize a matrix of pf16 values to float32.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix to dequantize.
    M : int
        The number of rows in the matrix.
    N : int
        The number of columns in the matrix.

    Returns
    -------
    numpy.ndarray (dtype=np.float32)
    """
    M = matrix.shape[0]
    N = matrix.shape[1]

    dequantized_matrix = np.zeros((M, N * 2), dtype=np.float32)

    for i in range(M):
        for j in range(0, N, 1):
            packed = matrix[i, j]
            float2 = np.float32(packed >> 16)
            float1 = np.float32(packed & 0xFFFF) 

            dequantized_matrix[i, j * 2] = float1
            dequantized_matrix[i, j * 2 + 1] = float2

    return dequantized_matrix


x = np.array([[1.0, 2.0, 3.0, 3.0],[4.0, 5.0, 6.0, 6.0]], dtype=np.float32)
print(x)
q = pf16(x)
print(q)
dq = dequant_pf16(q)
print(dq)
