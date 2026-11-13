// Copyright © 2023 Apple Inc.

#pragma once
#include "mlx/array.h"
#include "mlx/backend/common/ternary.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/threading.h"

namespace mlx::core {

template <typename T1, typename T2, typename T3, typename U, typename Op, int D>
void ternary_op_dims(
    const T1* a,
    const T2* b,
    const T3* c,
    U* out,
    Op op,
    const Shape& shape,
    const Strides& a_strides,
    const Strides& b_strides,
    const Strides& c_strides,
    const Strides& out_strides,
    int axis) {
  auto stride_a = a_strides[axis];
  auto stride_b = b_strides[axis];
  auto stride_c = c_strides[axis];
  auto stride_out = out_strides[axis];
  auto N = shape[axis];

  for (int i = 0; i < N; i++) {
    if constexpr (D > 1) {
      ternary_op_dims<T1, T2, T3, U, Op, D - 1>(
          a,
          b,
          c,
          out,
          op,
          shape,
          a_strides,
          b_strides,
          c_strides,
          out_strides,
          axis + 1);
    } else {
      *out = op(*a, *b, *c);
    }
    a += stride_a;
    b += stride_b;
    c += stride_c;
    out += stride_out;
  }
}

template <typename T1, typename T2, typename T3, typename U, typename Op>
void ternary_op_dispatch_dims(
    const T1* a_ptr,
    const T2* b_ptr,
    const T3* c_ptr,
    U* out_ptr,
    Op op,
    size_t size,
    Shape& shape,
    std::vector<Strides>& strides) {
  const auto& a_strides = strides[0];
  const auto& b_strides = strides[1];
  const auto& c_strides = strides[2];
  const auto& out_strides = strides[3];
  int ndim = shape.size();
  switch (ndim) {
    case 1:
      ternary_op_dims<T1, T2, T3, U, Op, 1>(
          a_ptr,
          b_ptr,
          c_ptr,
          out_ptr,
          op,
          shape,
          a_strides,
          b_strides,
          c_strides,
          out_strides,
          0);
      return;
    case 2:
      ternary_op_dims<T1, T2, T3, U, Op, 2>(
          a_ptr,
          b_ptr,
          c_ptr,
          out_ptr,
          op,
          shape,
          a_strides,
          b_strides,
          c_strides,
          out_strides,
          0);
      return;
  }

  ContiguousIterator a_it(shape, a_strides, ndim - 2);
  ContiguousIterator b_it(shape, b_strides, ndim - 2);
  ContiguousIterator c_it(shape, c_strides, ndim - 2);
  auto stride = out_strides[ndim - 3];
  for (size_t elem = 0; elem < size; elem += stride) {
    ternary_op_dims<T1, T2, T3, U, Op, 2>(
        a_ptr + a_it.loc,
        b_ptr + b_it.loc,
        c_ptr + c_it.loc,
        out_ptr + elem,
        op,
        shape,
        a_strides,
        b_strides,
        c_strides,
        out_strides,
        ndim - 2);
    a_it.step();
    b_it.step();
    c_it.step();
  }
}

template <typename T1, typename T2, typename T3, typename U, typename Op>
void ternary_op(
    const array& a,
    const array& b,
    const array& c,
    array& out,
    Op op,
    TernaryOpType topt) {
  const T1* a_ptr = a.data<T1>();
  const T2* b_ptr = b.data<T2>();
  const T3* c_ptr = c.data<T3>();
  U* out_ptr = out.data<U>();

  if (topt == TernaryOpType::ScalarScalarScalar) {
    *out_ptr = op(*a_ptr, *b_ptr, *c_ptr);
  } else if (topt == TernaryOpType::VectorVectorVector) {
    cpu::parallel_for(
        out.size(),
        cpu::elementwise_grain<U>(),
        [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_ptr[i] = op(a_ptr[i], b_ptr[i], c_ptr[i]);
          }
        });
  } else {
    auto [shape, strides] = collapse_contiguous_dims(
        a.shape(), {a.strides(), b.strides(), c.strides(), out.strides()});
    // Split the outermost (collapsed) axis across threads.
    int64_t outer = shape[0];
    int64_t inner_size = outer > 0 ? out.size() / outer : 0;
    int64_t grain = std::max<int64_t>(
        cpu::elementwise_grain<U>() / std::max<int64_t>(inner_size, 1), 1);
    cpu::parallel_for(outer, grain, [&](int64_t begin, int64_t end) {
      Shape shape_ = shape;
      shape_[0] = end - begin;
      ternary_op_dispatch_dims<T1, T2, T3, U>(
          a_ptr + begin * strides[0][0],
          b_ptr + begin * strides[1][0],
          c_ptr + begin * strides[2][0],
          out_ptr + begin * strides[3][0],
          op,
          (end - begin) * inner_size,
          shape_,
          strides);
    });
  }
}

} // namespace mlx::core
