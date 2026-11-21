# encoding: utf-8
#
# test_convolution_matrix.py
#
# Tests of convolution_matrix (LSF matching of the fiber flats, flux calibration)

import numpy as np
import pytest
from scipy import sparse

from lvmdrp.core.spectrum1d import Spectrum1D, convolution_matrix


def _reference(kernel, normalize=True):
    """ convolution_matrix as it was before it was vectorized """
    if len(kernel.shape) == 2:
        nrows = kernel.shape[0]
    elif len(kernel.shape) == 1:
        nrows = kernel.size
        kernel = np.repeat([kernel], nrows, axis=0)
    kernelLength = kernel.shape[1]
    rowIdxFirst = np.floor(kernelLength / 2)
    rowIdxLast = rowIdxFirst + nrows
    vI, vJ, vV = [], [], []
    for jj in range(nrows):
        kernel_row = kernel[jj] / np.sum(kernel[jj])
        for ii in range(kernelLength):
            if (ii + jj >= rowIdxFirst) and (ii + jj < rowIdxLast):
                vI.append(int(ii + jj - rowIdxFirst))
                vJ.append(int(jj))
                vV.append(kernel_row[ii])
    new_kernel = sparse.csr_array((vV, (vJ, vI)))
    if normalize:
        new_kernel = new_kernel.multiply(1 / np.sum(new_kernel, axis=1))
    return new_kernel


def _gaussian_kernels(sigmas):
    pixels = np.arange(-np.ceil(3 * sigmas.max()), np.ceil(3 * sigmas.max()))
    return np.exp(-0.5 * (pixels[None, :] / sigmas[:, None]) ** 2)


@pytest.mark.parametrize('normalize', [True, False])
@pytest.mark.parametrize('sigmas', [np.linspace(0.8, 2.5, 300), np.full(50, 1.3), np.linspace(3.0, 0.6, 7)])
def test_identical_to_reference(sigmas, normalize):
    kernel = _gaussian_kernels(sigmas)
    new, ref = convolution_matrix(kernel, normalize=normalize), _reference(kernel, normalize=normalize)
    assert new.shape == ref.shape
    new, ref = sparse.csr_array(new), sparse.csr_array(ref)
    assert np.array_equal(new.indptr, ref.indptr)
    assert np.array_equal(new.indices, ref.indices)
    assert np.array_equal(new.data, ref.data)


def test_identical_to_reference_1d_kernel():
    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 1.7) ** 2)
    new, ref = sparse.csr_array(convolution_matrix(kernel)), sparse.csr_array(_reference(kernel))
    assert np.array_equal(new.toarray(), ref.toarray())


def test_flatten_lsf_degrades_to_target():
    rng = np.random.default_rng(3)
    wave = np.arange(5000.0, 5400.0, 0.5)
    lsf = np.linspace(2.0, 3.0, wave.size)
    data = 100 + rng.normal(0, 1, wave.size)
    spec = Spectrum1D(wave=wave, data=data, error=np.ones_like(data), lsf=lsf, mask=np.zeros(wave.size, dtype=bool))
    new = spec.flatten_lsf(4.5)
    assert np.allclose(new._lsf, 4.5)
    # a flat spectrum stays flat away from the edges, and the noise drops
    assert np.isclose(np.mean(new._data[50:-50]), 100, atol=0.2)
    assert np.std(new._data[50:-50]) < 0.5 * np.std(data[50:-50])
