import numpy as np
from numba import jit
from enum import Enum
from tqdm import tqdm
from iteration_utilities import grouper

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

    quantized_matrix = []
    matrix = matrix.astype(np.float16)
    matrix = matrix.view(np.uint16)
    matrix = matrix.flatten()

    for chunk in grouper(matrix, 2):
        float0 = chunk[0]
        float1 = chunk[1]
        packed_value = ((np.uint32(float1)) << 16) | np.uint32(float0)
        quantized_matrix.append(packed_value)

    quantized_matrix = np.array(quantized_matrix, dtype=np.uint32)
    return quantized_matrix.reshape((M, N // 2))


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
            half1 = np.array(packed & 0xFFFF, dtype=np.uint16).view(np.float16)
            half2 = np.array((packed >> 16) & 0xFFFF, dtype=np.uint16).view(np.float16)
            float1 = np.float32(half1)    # Convert to float32
            float2 = np.float32(half2)    # Convert to float32
            dequantized_matrix[i, j * 2] = float1
            dequantized_matrix[i, j * 2 + 1] = float2

    return dequantized_matrix


def validate():
    x = np.array(
        [
            [0.1, -0.1, 0.5, -0.5],
            [1.0, -1.0, 1.2, -1.2],
            [0.1, -0.1, 0.5, -0.5],
            [1.0, -1.0, 1.2, -1.2],
        ],
        dtype=np.float32,
    )
    print("Before Quant: \n", x)
    q = pf16(x)
    print("After Quant: \n", q)
    dq = dequant_pf16(q)
    print("After Dequant: \n", dq)


validate()
