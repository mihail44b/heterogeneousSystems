#pragma once

#include <cuda_runtime.h>

#if !defined(__CUDACC__)

namespace nvcuda::wmma {

enum layout_t { mem_row_major, mem_col_major };

struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};

template <typename Use, int m, int n, int k, typename T, typename Layout = void>
struct fragment {};

template <typename Frag, typename T>
inline void fill_fragment(Frag&, T) {}

template <typename Frag, typename T>
inline void load_matrix_sync(Frag&, const T*, unsigned) {}

template <typename FragC, typename FragA, typename FragB>
inline void mma_sync(FragC&, const FragA&, const FragB&, const FragC&) {}

template <typename Frag, typename T>
inline void store_matrix_sync(T*, const Frag&, unsigned, layout_t) {}

}  // namespace nvcuda::wmma

#endif
