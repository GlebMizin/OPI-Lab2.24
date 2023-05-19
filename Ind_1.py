#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_array_equal
from threading import Thread
from time import time


# Разобьём двумерный массив arr на блоки размером nrows и ncols
def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    n, m = h // nrows, w // ncols
    return arr.reshape(nrows, n, ncols, m).swapaxes(1, 2)


# Произведём умножение матриц стандартным методом и запишем в out
def original_dot_func(a, b, out):
    out[:] = np.dot(a, b)


# Умножаем матрицы a и b, используя nblocks и mblocks потоков.
def parallel_dot(a, b, nblocks, mblocks, original_dot_func):
    n_jobs = nblocks * mblocks
    print(f'Running {n_jobs} jobs in parallel')

    out = np.empty((a.shape[0], b.shape[1]), dtype=a.dtype)

    # Разбиваем матрицы на блоки функцией blockshaped
    out_blocks = blockshaped(out, nblocks, mblocks)
    a_blocks = blockshaped(a, nblocks, 1)
    b_blocks = blockshaped(b, 1, mblocks)

    """Умножаем каждый блок функцией original_dot
    После чего возвращаем выходной массив вызывающей программе"""

    threads = []
    for i in range(nblocks):
        for j in range(mblocks):
            th = Thread(target=original_dot_func,
                        args=(a_blocks[i, 0, :, :],
                              b_blocks[0, j, :, :],
                              out_blocks[i, j, :, :])
                        )
            th.start()
            threads.append(th)

    for th in threads:
        th.join()

    return out


if __name__ == '__main__':

    # Проверяем затраченное время на умножения разными способами
    a = np.ones((4, 3), dtype=int)
    b = np.arange(18, dtype=int).reshape(3, 6)
    assert_array_equal(parallel_dot(a, b, 2, 2, original_dot_func), np.dot(a, b))

    a = np.random.randn(1500, 1500).astype(int)

    start = time()
    parallel_dot(a, a, 2, 4, original_dot_func)
    time_par = time() - start
    print('Параллельное умножение матриц заняло:  {:.2f} секунд '
          .format(time_par)
          )

    start = time()
    np.dot(a, a)
    time_dot = time() - start
    print('Перемножение матриц функцией np.dot заняло: {:.2f} секунд'
          .format(time_dot)
          )
