// This code is a Qiskit project.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::borrow::Cow;

use ndarray::{ArrayView1, ArrayViewMut2};
use numpy::Complex64;
use numpy::PyReadonlyArray1;
use numpy::PyReadwriteArray2;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Minimum number of modified matrix entries for the phase shift to run in
/// parallel. Below this, the fixed cost of dispatching to the thread pool
/// dominates.
const PARALLEL_MIN_ENTRIES: usize = 1 << 16;

/// Sets the smallest unit of parallel work: the rows are divided into this many
/// blocks per thread. Rayon splits work adaptively and never divides a block, so
/// this is a lower bound on task size rather than a fixed schedule.
const BLOCKS_PER_THREAD: usize = 16;

/// Apply a phase shift to a submatrix of a matrix in-place.
///
/// Multiplies ``vec[i, j]`` by ``phase`` for every ``i`` in ``row_indices`` and
/// every ``j`` in ``col_indices``. Passing ``None`` for either selects all rows or
/// all columns. The indices must not contain duplicates: if they do, the result
/// is unspecified, since the serial path applies the phase once per occurrence
/// while the parallel path applies it only once for a repeated row index.
#[pyfunction]
#[pyo3(signature = (vec, phase, row_indices=None, col_indices=None))]
pub fn apply_phase_shift_in_place(
    mut vec: PyReadwriteArray2<Complex64>,
    phase: Complex64,
    row_indices: Option<PyReadonlyArray1<usize>>,
    col_indices: Option<PyReadonlyArray1<usize>>,
) {
    let mut vec = vec.as_array_mut();
    let (n_rows, n_cols) = vec.dim();
    let row_indices = row_indices
        .as_ref()
        .map(|indices| to_slice(indices.as_array()));
    let col_indices = col_indices
        .as_ref()
        .map(|indices| to_slice(indices.as_array()));
    let row_indices = row_indices.as_deref();
    let col_indices = col_indices.as_deref();

    let Some(data) = vec.as_slice_mut() else {
        // Not C-contiguous, so rows can't be taken as slices of the buffer
        match row_indices {
            None => scale_rows_strided(&mut vec, 0..n_rows, col_indices, phase),
            Some(rows) => scale_rows_strided(&mut vec, rows.iter().copied(), col_indices, phase),
        }
        return;
    };
    if data.is_empty() {
        return;
    }

    let parallel = use_parallel(n_rows, n_cols, row_indices, col_indices);

    match col_indices {
        None if n_cols == 1 => match row_indices {
            None => data.iter_mut().for_each(|x| *x *= phase),
            Some(row_indices) => row_indices.iter().for_each(|&i| data[i] *= phase),
        },
        None => scale_rows(data, n_rows, n_cols, row_indices, parallel, |row| {
            row.iter_mut().for_each(|x| *x *= phase)
        }),
        Some(col_indices) => scale_rows(data, n_rows, n_cols, row_indices, parallel, |row| {
            col_indices.iter().for_each(|&j| row[j] *= phase)
        }),
    }
}

/// Whether the phase shift should run in parallel.
fn use_parallel(
    n_rows: usize,
    n_cols: usize,
    row_indices: Option<&[usize]>,
    col_indices: Option<&[usize]>,
) -> bool {
    let row_len = col_indices.map_or(n_cols, <[usize]>::len);
    let n_entries = row_indices.map_or(n_rows, <[usize]>::len) * row_len;
    // A single-column matrix (the spinless case) was measured to be faster serially
    // at every size up to millions of entries. Narrow rows selected from a wider
    // matrix are scattered gathers, which do benefit from parallelism.
    rayon::current_num_threads() > 1 && n_cols > 1 && n_entries >= PARALLEL_MIN_ENTRIES
}

/// Apply a phase shift to the given rows of a matrix that is not C-contiguous.
///
/// Generic over the row iterator so that all rows and selected rows share one
/// implementation without dynamic dispatch.
fn scale_rows_strided(
    vec: &mut ArrayViewMut2<Complex64>,
    rows: impl Iterator<Item = usize>,
    col_indices: Option<&[usize]>,
    phase: Complex64,
) {
    for i in rows {
        let mut row = vec.row_mut(i);
        match col_indices {
            None => row.map_inplace(|x| *x *= phase),
            Some(col_indices) => col_indices.iter().for_each(|&j| row[j] *= phase),
        }
    }
}

/// Apply ``scale_row`` to the selected rows of a C-contiguous matrix.
///
/// Generic over the row operation so that it is specialized for each caller.
fn scale_rows<F>(
    data: &mut [Complex64],
    n_rows: usize,
    n_cols: usize,
    row_indices: Option<&[usize]>,
    parallel: bool,
    scale_row: F,
) where
    F: Fn(&mut [Complex64]) + Sync,
{
    if !parallel {
        match row_indices {
            None => data.chunks_mut(n_cols).for_each(scale_row),
            Some(row_indices) => row_indices
                .iter()
                .for_each(|&i| scale_row(&mut data[i * n_cols..(i + 1) * n_cols])),
        }
        return;
    }

    // Contiguous blocks of rows, which bound how finely rayon can split the work
    let rows_per_block = n_rows.div_ceil(rayon::current_num_threads() * BLOCKS_PER_THREAD);
    let blocks = data.par_chunks_mut(rows_per_block * n_cols);
    match row_indices {
        None => blocks.for_each(|block| block.chunks_mut(n_cols).for_each(&scale_row)),
        Some(row_indices) => {
            // Rows are visited through a mask so that the selected rows can be
            // updated in parallel without aliasing mutable borrows
            let mut mask = vec![false; n_rows];
            row_indices.iter().for_each(|&i| mask[i] = true);
            blocks
                .zip(mask.par_chunks(rows_per_block))
                .for_each(|(block, mask)| {
                    block
                        .chunks_mut(n_cols)
                        .zip(mask)
                        .filter(|(_, &selected)| selected)
                        .for_each(|(row, _)| scale_row(row))
                });
        }
    }
}

/// Borrow the array as a slice if it is contiguous, otherwise copy it.
fn to_slice(array: ArrayView1<'_, usize>) -> Cow<'_, [usize]> {
    match array.to_slice() {
        Some(slice) => Cow::Borrowed(slice),
        None => Cow::Owned(array.to_vec()),
    }
}
