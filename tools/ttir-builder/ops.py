# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Callable, Dict, Any
from ttmlir.ir import *
from ttmlir.dialects import ttir, ttcore, tensor, quant
from ttmlir.passes import GoldenTensor, DataType
import torch
from enum import Enum, auto
import re
from .ccl_golden import *
from sphinx.ext.autodoc import FunctionDocumenter

# Alias for operands of ops which can be either BlockArguments, Values, or other
# ops wrapped in OpView or Operation.
Operand = Union[Value, OpView, Operation]

# Convenience alias for shape
Shape = Union[List[int], Tuple[int, ...]]


class TTIRBuilderOps:

    # TTIR top level ops

    def get_dimension_size(
        self, in0: Operand, dimension: int = 0, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.get_dimension_size``.

        *Dimension size query operation.*

        Returns the size of the specified dimension of the input tensor.

        .. code-block:: mlir

            // Get size of dimension 0 from input tensor
            %result = ttir.get_dimension_size(%input) {
              dimension = 0
            } : tensor<3x2x7xf32> -> tensor<i32>
            // Result: tensor<i32> with value [3]

        Parameters
        ----------
        in0 : Operand
            Input tensor operand to get dimension size from
        dimension : int, optional
            The dimension index to get size of (default: 0)
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        golden_data = [self._get_golden_tensor(in0).size(dimension)]
        return self.op_proxy(
            torch.tensor,
            ttir.GetDimensionSizeOp,
            [in0],
            golden_kwargs={"data": golden_data, "dtype": torch.int32},
            ttir_kwargs={"dimension": dimension},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0]),
            organize_golden_args=lambda i: 0,
            output_type=self.get_type_from_torch_dtype(torch.int32),
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def dot_general(
        self,
        in0: Operand,
        in1: Operand,
        out0: Operand,
        batch_dims_lhs: List[int],
        contract_dims_lhs: List[int],
        batch_dims_rhs: List[int],
        contract_dims_rhs: List[int],
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.dot_general``.

        *Generalized dot product operation.*

        A flexible tensor operation that generalizes matrix multiplication by allowing user to specify which
        dimensions of two tensors to contract. Matrix multiplication is a special case of this operation,
        where the contraction happens along the last axis of the first tensor and the second-to-last axis
        of the second tensor.

        Based on StableHLO DotGeneral Op (https://openxla.org/stablehlo/spec#dot_general)

        .. code-block:: mlir

            // Matrix multiplication example
            %result = ttir.dot_general(%lhs, %rhs, %out) {
              batch_dims_lhs = [],
              contract_dims_lhs = [1],
              batch_dims_rhs = [],
              contract_dims_rhs = [0]
            } : tensor<2x3xf32>, tensor<3x4xf32>, tensor<2x4xf32> -> tensor<2x4xf32>

            // Batched matrix multiplication
            %result = ttir.dot_general(%lhs, %rhs, %out) {
              batch_dims_lhs = [0],
              contract_dims_lhs = [2],
              batch_dims_rhs = [0],
              contract_dims_rhs = [1]
            } : tensor<10x2x3xf32>, tensor<10x4x3xf32>, tensor<10x2x4xf32> -> tensor<10x2x4xf32>

        Parameters
        ----------
        in0 : Operand
            Left-hand side input tensor
        in1 : Operand
            Right-hand side input tensor
        out0 : Operand
            Output tensor
        batch_dims_lhs : List[int]
            Batch dimensions for the left-hand side tensor
        contract_dims_lhs : List[int]
            Contracting dimensions for the left-hand side tensor
        batch_dims_rhs : List[int]
            Batch dimensions for the right-hand side tensor
        contract_dims_rhs : List[int]
            Contracting dimensions for the right-hand side tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        kwargs = {
            "batch_dims_lhs": batch_dims_lhs,
            "contract_dims_lhs": contract_dims_lhs,
            "batch_dims_rhs": batch_dims_rhs,
            "contract_dims_rhs": contract_dims_rhs,
        }
        return self.op_proxy(
            self.dot_general_golden_function,
            ttir.DotGeneralOp,
            [in0, in1, out0],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1]),
            output_type=self.get_type_from_torch_dtype(
                self._get_golden_tensor(in0).dtype
            ),
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def dot_general_golden_function(
        self,
        lhs,
        rhs,
        out,
        batch_dims_lhs,
        contract_dims_lhs,
        batch_dims_rhs,
        contract_dims_rhs,
    ):
        non_batch_dims_lhs = [d for d in range(lhs.dim()) if d not in batch_dims_lhs]
        non_batch_dims_rhs = [d for d in range(rhs.dim()) if d not in batch_dims_rhs]
        transposed_lhs = torch.permute(lhs, (batch_dims_lhs + non_batch_dims_lhs))
        transposed_rhs = torch.permute(rhs, (batch_dims_rhs + non_batch_dims_rhs))
        result_batching_dims = list(range(len(batch_dims_lhs)))
        result = torch.empty(*out.shape, dtype=lhs.dtype)

        dim_ranges = []
        for i in range(len(result_batching_dims)):
            dim_ranges.append([j for j in range(list(lhs.shape)[i])])
        import itertools

        batch_indices = list(itertools.product(*dim_ranges))
        for index in batch_indices:
            transposed_lhs_slice = transposed_lhs[index]
            transposed_rhs_slice = transposed_rhs[index]
            dot_dims_lhs = [d - len(index) for d in contract_dims_lhs]
            dot_dims_rhs = [d - len(index) for d in contract_dims_rhs]
            out_index = index
            result[out_index] = torch.tensordot(
                transposed_lhs_slice,
                transposed_rhs_slice,
                dims=(dot_dims_lhs, dot_dims_rhs),
            )
        return result

    # TTIR top level named ops
    # class TTIR_ElementwiseTernaryOp

    def where(
        self,
        in0: Operand,
        in1: Operand,
        in2: Operand,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.where``.

        *Elementwise conditional selection operation.*

        For each element position, selects between two values based on a boolean condition:
        - If the condition is true (non-zero), selects from the first value tensor
        - If the condition is false (zero), selects from the second value tensor

        Supports broadcasting according to standard broadcasting rules.

        .. code-block:: mlir

            // Basic selection between two tensors
            %result = ttir.where(%cond, %true_vals, %false_vals) :
                tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
            // Input tensors:
            // %cond: [[1, 0], [0, 1]]
            // %true_vals: [[1.0, 2.0], [3.0, 4.0]]
            // %false_vals: [[5.0, 6.0], [7.0, 8.0]]
            // Output tensor:
            // [[1.0, 6.0], [7.0, 4.0]]

            // With broadcasting (scalar condition)
            %result = ttir.where(%scalar_cond, %true_vals, %false_vals) :
                tensor<i1>, tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>

        Parameters
        ----------
        in0 : Operand
            Condition tensor (predicate)
        in1 : Operand
            Tensor containing values to select when condition is true
        in2 : Operand
            Tensor containing values to select when condition is false
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        # Handle golden condition tensor
        in0_tensor = self._get_golden_tensor(in0)
        condition = torch.full(in0_tensor.shape, False)
        condition[in0_tensor > 0] = True
        return self.op_proxy(
            torch.where,
            ttir.WhereOp,
            [in0, in1, in2],
            organize_golden_args=lambda i: (
                condition,
                self._get_golden_tensor(i[1]),
                self._get_golden_tensor(i[2]),
            ),
            unit_attrs=unit_attrs,
        )

    # class TTIR_ElementwiseUnaryOp

    # class TTIR_ElementwiseUnaryOp

    def abs(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.abs``.

        *Elementwise absolute value operation.*

        Computes the absolute value of each element in the input tensor.

        .. code-block:: mlir

            // Compute absolute values of all elements in %input
            %result = ttir.abs(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
            // Input tensor:
            // [[-2.5,  3.7,  0.0,  1.2], ... ]
            // Output tensor:
            // [[2.5, 3.7, 0.0, 1.2], ... ]

            // Example with integer tensor
            %result = ttir.abs(%int_input, %int_output) : tensor<10xi32>, tensor<10xi32> -> tensor<10xi32>
            // Input tensor:
            // [-5, 0, 3, -2, ...]
            // Output tensor:
            // [5, 0, 3, 2, ...]

        Parameters
        ----------
        in0 : Operand
            Input tensor to compute absolute value of
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self.eltwise_proxy(torch.abs, ttir.AbsOp, [in0], unit_attrs)

    def cbrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.cbrt``.

        *Elementwise cubic root operation.*

        Computes the cubic root (∛) of each element in the input tensor.
        For each element, returns the real-valued number that, when cubed, equals the input value.
        Unlike square root, cubic root is defined for negative numbers as well as positive numbers.

        .. code-block:: mlir

            // Compute cubic root of all elements
            %result = ttir.cbrt(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [8.0, 27.0, -8.0, 1.0]
            // Output tensor:
            // [2.0, 3.0, -2.0, 1.0]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            A tensor containing the cubic root of each element in the input tensor
        """
        golden = self._get_golden_tensor(in0)
        golden_sign = torch.sign(golden)
        golden_cbrt = torch.pow(torch.abs(golden), 1 / 3)
        return self.op_proxy(
            torch.mul,
            ttir.CbrtOp,
            [in0],
            golden_kwargs={"input": golden_sign, "other": golden_cbrt},
            organize_golden_args=lambda i: 0,
            unit_attrs=unit_attrs,
        )

    def gelu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.gelu``.

        *Elementwise GELU operation.*

        Computes the GELU (Gaussian Error Linear Unit) of each element in the input tensor.
        GELU is a smooth, non-monotonic activation function that approximates the cumulative
        distribution function of a standard normal distribution.

        Mathematical definition: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

        .. code-block:: mlir

            // Compute GELU of all elements
            %result = ttir.gelu(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [1.0, -0.5, 2.0, -2.0]
            // Output tensor:
            // [0.841, -0.154, 1.954, -0.046]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            A tensor containing the GELU values of each element in the input tensor
        """
        return self.eltwise_proxy(torch.gelu, ttir.GeluOp, [in0], unit_attrs)

    def cos(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.cos``.

        *Elementwise cosine operation.*

        Computes the cosine of each element in the input tensor.
        Input values are expected to be in radians.

        .. code-block:: mlir

            // Compute cosine of all elements
            %result = ttir.cos(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor (in radians):
            // [0.0, 3.14159, 1.5708, -1.5708]
            // Output tensor:
            // [1.0, -1.0, 0.0, 0.0]

        Parameters
        ----------
        in0 : Operand
            Input tensor (values in radians)
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            A tensor containing the cosine of each element in the input tensor
        """
        return self.eltwise_proxy(torch.cos, ttir.CosOp, [in0], unit_attrs)

    def log1p(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise natural logarithm of one plus input operation.

        The `log1p` operation computes the natural logarithm of one plus each element in the
        input tensor. For each element x, it returns ln(1 + x). This operation is more
        accurate than computing log(1 + x) directly for x values close to zero, and it is
        defined for x > -1. For values less than or equal to -1, the behavior depends on
        the implementation (may return NaN or negative infinity).

        .. code-block:: mlir

            // Compute log1p of all elements
            %result = ttir.log1p(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [0.0, -0.999, 7.0, 6.38905621, 15.0]
            // Output tensor:
            // [0.0, -6.90776825, 2.07944155, 2.0, 2.77258873]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            A tensor containing the log1p values of the input tensor
        """
        return self.eltwise_proxy(torch.log1p, ttir.Log1pOp, [in0], unit_attrs)

    def is_finite(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.is_finite``.

        *Elementwise finite check operation.*

        Checks if each element in the input tensor is finite (neither infinite nor NaN).
        For each element, returns a boolean value indicating whether the element is finite.

        Mathematical definition: isfinite(x) = x ∈ ℝ

        .. code-block:: mlir

            // Check if elements are finite
            %result = ttir.is_finite(%input, %output) : tensor<4xf32>, tensor<4xi1> -> tensor<4xi1>
            // Input tensor:
            // [1.0, inf, -inf, nan]
            // Output tensor:
            // [true, false, false, false]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        return self.eltwise_proxy(torch.isfinite, ttir.IsFiniteOp, [in0], unit_attrs)

    # class TTIR_ElementwiseUnaryWithFloatParameterOp

    def leaky_relu(
        self,
        in0: Operand,
        parameter: float = 0.01,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Elementwise leaky ReLU operation.

        *Elementwise leaky ReLU activation operation.*

        Computes an element-wise activation function over its input tensor.
        For each element x, computes:
        - y = x if x > 0
        - y = parameter * x if x <= 0

        The parameter is a small, user-defined constant that determines the slope for
        negative inputs.

        .. code-block:: mlir

            // Compute Leaky ReLU with default parameter (0.01)
            %result = ttir.leaky_relu(%input) {
              parameter = 0.01
            } : tensor<2x2xf32> -> tensor<2x2xf32>
            // Input tensor:
            // [[1.0, -2.0],
            //  [3.0, -4.0]]
            // Output tensor:
            // [[1.0, -0.02],
            //  [3.0, -0.04]]

            // With custom parameter
            %result = ttir.leaky_relu(%input) {
              parameter = 0.1
            } : tensor<2x2xf32> -> tensor<2x2xf32>
            // Input tensor:
            // [[1.0, -2.0],
            //  [3.0, -4.0]]
            // Output tensor:
            // [[1.0, -0.2],
            //  [3.0, -0.4]]

        Parameters
        ----------
        in0 : Operand
            Input tensor to be activated
        parameter : float, optional
            The slope for negative values (default: 0.01)
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        # TODO: reconcile this naming mismatch
        ttir_kwargs = {"parameter": parameter}
        golden_kwargs = {"negative_slope": parameter}
        return self.op_proxy(
            torch.nn.functional.leaky_relu,
            ttir.LeakyReluOp,
            [in0],
            golden_kwargs=golden_kwargs,
            ttir_kwargs=ttir_kwargs,
            unit_attrs=unit_attrs,
        )

    # class TTIR_ElementwiseBinaryOp

    def div(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.div``.

        *Elementwise division operation.*

        Performs elementwise division between two tensors.
        For each pair of corresponding elements, divides the element in the first
        tensor by the element in the second tensor.

        Note: Division by zero behavior depends on the implementation and data type.

        Mathematical definition: div(x, y) = x / y

        .. code-block:: mlir

            // Divide corresponding elements
            %result = ttir.div(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensors:
            // lhs: [3.5, 0.0, -1.2]
            // rhs: [1.5, 2.0, -3.2]
            // Output tensor:
            // [2.333, 0.0, 0.375]

        Parameters
        ----------
        in0 : Operand
            First input tensor (dividend)
        in1 : Operand
            Second input tensor (divisor)
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            A tensor containing the elementwise quotient of the inputs
        """
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.div,
            ttir.DivOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
            unit_attrs=unit_attrs,
        )

    def eq(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.eq``.

        *Elementwise equality comparison operation.*

        Performs an elementwise equality comparison between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if the elements are equal
        - 0 (false) if the elements are not equal

        Note that special handling may be required for floating-point NaN values, as NaN is not
        equal to any value, including itself.

        Mathematical definition: equal(x, y) = x == y

        .. code-block:: mlir

            // Compare elements for equality
            %result = ttir.eq(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xi1> -> tensor<3xi1>
            // Input tensors:
            // lhs: [1.0, 2.0, 3.0]
            // rhs: [1.0, 2.0, 4.0]
            // Output tensor:
            // [1, 1, 0]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.eq,
            ttir.EqualOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
            unit_attrs=unit_attrs,
        )

    def ne(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.ne``.

        *Elementwise inequality comparison operation.*

        Performs elementwise inequality comparison between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if the elements are not equal
        - 0 (false) if the elements are equal

        Note: Special handling may be required for floating-point NaN values, as NaN is not
        equal to any value, including itself. This means ne(NaN, NaN) should return true.

        Mathematical definition: not_equal(x, y) = x != y

        .. code-block:: mlir

            // Compare elements for inequality
            %result = ttir.ne(%lhs, %rhs, %output) : tensor<4xf32>, tensor<4xf32>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1.0, 2.0, 3.0, 2.0]
            // rhs: [1.0, 2.0, 4.0, 5.0]
            // Output tensor:
            // [0, 0, 1, 1]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        return self.op_proxy(
            torch.ne,
            ttir.NotEqualOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def ge(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.ge``.

        *Elementwise greater than or equal to comparison operation.*

        Performs elementwise greater than or equal to comparison between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if the left element is greater than or equal to the right element
        - 0 (false) if the left element is less than the right element

        Mathematical definition: greater_equal(x, y) = x >= y

        .. code-block:: mlir

            // Compare elements for greater than or equal to
            %result = ttir.ge(%lhs, %rhs, %output) : tensor<4xf32>, tensor<4xf32>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1.0, 2.0, 3.0, 2.0]
            // rhs: [1.0, 2.0, 4.0, 5.0]
            // Output tensor:
            // [1, 1, 0, 0]

        Parameters
        ----------
        in0 : Operand
            First input tensor (left-hand side)
        in1 : Operand
            Second input tensor (right-hand side)
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.ge,
            ttir.GreaterEqualOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
            unit_attrs=unit_attrs,
        )

    def gt(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.gt``.

        *Elementwise greater than comparison operation.*

        Performs elementwise greater than comparison between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if the left element is greater than the right element
        - 0 (false) if the left element is less than or equal to the right element

        Mathematical definition: greater(x, y) = x > y

        .. code-block:: mlir

            // Compare elements for greater than
            %result = ttir.gt(%lhs, %rhs, %output) : tensor<4xf32>, tensor<4xf32>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1.0, 2.0, 3.0, 2.0]
            // rhs: [1.0, 1.0, 4.0, 5.0]
            // Output tensor:
            // [0, 1, 0, 0]

        Parameters
        ----------
        in0 : Operand
            First input tensor (left-hand side)
        in1 : Operand
            Second input tensor (right-hand side)
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.gt,
            ttir.GreaterThanOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
            unit_attrs=unit_attrs,
        )

    def le(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.le``.

        *Elementwise less than or equal to comparison operation.*

        Performs elementwise less than or equal to comparison between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if the left element is less than or equal to the right element
        - 0 (false) if the left element is greater than the right element

        Mathematical definition: less_equal(x, y) = x <= y

        .. code-block:: mlir

            // Compare elements for less than or equal to
            %result = ttir.le(%lhs, %rhs, %output) : tensor<4xf32>, tensor<4xf32>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1.0, 2.0, 3.0, 2.0]
            // rhs: [1.0, 2.0, 4.0, 5.0]
            // Output tensor:
            // [1, 1, 1, 1]

        Parameters
        ----------
        in0 : Operand
            First input tensor (left-hand side)
        in1 : Operand
            Second input tensor (right-hand side)
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.le,
            ttir.LessEqualOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
            unit_attrs=unit_attrs,
        )

    def lt(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.lt``.

        *Elementwise less than comparison operation.*

        The `lt` operation performs an elementwise less than comparison between two tensors.
        For each pair of corresponding elements, it returns:
        - 1 (true) if the left element is less than the right element
        - 0 (false) if the left element is greater than or equal to the right element

        Mathematical definition: less(x, y) = x < y

        .. code-block:: mlir

            // Compare elements for less than
            %result = ttir.lt(%lhs, %rhs, %output) : tensor<4xf32>, tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensors:
            // lhs: [1.0, 2.0, 3.0, 2.0]
            // rhs: [1.0, 2.0, 4.0, 5.0]
            // Output tensor: [0, 0, 1, 1]  # 1 where less, 0 where greater or equal

        Parameters
        ----------
        in0 : Operand
            First input tensor (left-hand side)
        in1 : Operand
            Second input tensor (right-hand side)
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            A boolean tensor with 1s where left < right and 0s otherwise
        """
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.lt,
            ttir.LessThanOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
            unit_attrs=unit_attrs,
        )

    def logical_and(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.logical_and``.

        *Elementwise logical AND operation.*

        Performs elementwise logical AND operation between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if both elements are 1 (true)
        - 0 (false) if at least one element is 0 (false)

        This operation is idempotent, meaning logical_and(x, x) = x.

        .. code-block:: mlir

            // Logical AND operation
            %result = ttir.logical_and(%lhs, %rhs, %output) : tensor<4xi1>, tensor<4xi1>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1, 0, 1, 0]
            // rhs: [1, 1, 0, 1]
            // Output tensor:
            // [1, 0, 0, 0]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.logical_and,
            ttir.LogicalAndOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
            unit_attrs=unit_attrs,
        )

    def logical_or(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.logical_or``.

        *Elementwise logical OR operation.*

        Performs elementwise logical OR operation between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if at least one element is 1 (true)
        - 0 (false) if both elements are 0 (false)

        This operation is idempotent, meaning logical_or(x, x) = x.

        Mathematical definition: logical_or(x, y) = x || y

        .. code-block:: mlir

            // Logical OR operation
            %result = ttir.logical_or(%lhs, %rhs, %output) : tensor<4xi1>, tensor<4xi1>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1, 0, 1, 0]
            // rhs: [1, 1, 0, 1]
            // Output tensor:
            // [1, 1, 1, 1]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.logical_or,
            ttir.LogicalOrOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
            unit_attrs=unit_attrs,
        )

    def logical_xor(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.logical_xor``.

        *Elementwise logical XOR operation.*

        Performs elementwise logical XOR (exclusive OR) operation between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if exactly one element is 1 (true)
        - 0 (false) if both elements are the same (both 0 or both 1)

        Mathematical definition: logical_xor(x, y) = (x || y) && !(x && y)

        .. code-block:: mlir

            // Logical XOR operation
            %result = ttir.logical_xor(%lhs, %rhs, %output) : tensor<4xi1>, tensor<4xi1>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1, 0, 1, 0]
            // rhs: [1, 1, 0, 1]
            // Output tensor:
            // [0, 1, 1, 1]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.logical_xor,
            ttir.LogicalXorOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def bitwise_and(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.bitwise_and``.

        *Elementwise bitwise AND operation.*

        Performs elementwise bitwise AND operation between two tensors.
        For each pair of corresponding elements, performs a bitwise AND on their binary representations.

        This operation is typically used with integer data types and has the following properties:
        - Commutative: bitwise_and(x, y) = bitwise_and(y, x)
        - Associative: bitwise_and(x, bitwise_and(y, z)) = bitwise_and(bitwise_and(x, y), z)
        - Identity: bitwise_and(x, -1) = x
        - Zero: bitwise_and(x, 0) = 0

        .. code-block:: mlir

            // Bitwise AND with integer tensors
            %result = ttir.bitwise_and(%lhs, %rhs, %output) : tensor<3xi8>, tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
            // Input tensors:
            // lhs: [5, 3, 255]  (binary: [00000101, 00000011, 11111111])
            // rhs: [3, 6, 129]   (binary: [00000011, 00000110, 10000001])
            // Output tensor:
            // [1, 2, 129]    (binary: [00000001, 00000010, 10000001])

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        return self.eltwise_proxy(
            torch.bitwise_and, ttir.BitwiseAndOp, [in0, in1], unit_attrs=unit_attrs
        )

    @autodoc_skip
    def bitwise_or(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.bitwise_or``.

        *Elementwise bitwise OR operation.*

        Performs elementwise bitwise OR operation between two tensors.
        For each pair of corresponding elements, performs a bitwise OR on their binary representations.

        This operation is typically used with integer data types and has the following properties:
        - Commutative: bitwise_or(x, y) = bitwise_or(y, x)
        - Associative: bitwise_or(x, bitwise_or(y, z)) = bitwise_or(bitwise_or(x, y), z)
        - Identity: bitwise_or(x, 0) = x
        - One: bitwise_or(x, -1) = -1

        .. code-block:: mlir

            // Bitwise OR with integer tensors
            %result = ttir.bitwise_or(%lhs, %rhs, %output) : tensor<3xi8>, tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
            // Input tensors:
            // lhs: [5, 3, 255]  (binary: [00000101, 00000011, 11111111])
            // rhs: [3, 6, 129]   (binary: [00000011, 00000110, 10000001])
            // Output tensor:
            // [7, 7, 255]    (binary: [00000111, 00000111, 11111111])

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        return self.eltwise_proxy(
            torch.bitwise_or, ttir.BitwiseOrOp, [in0, in1], unit_attrs=unit_attrs
        )

    @autodoc_skip
    def bitwise_not(
        self, in0: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.bitwise_not``.

        *Elementwise bitwise NOT operation.*

        Computes the bitwise NOT (one's complement) of each element in the input tensor.
        For each element, flips all the bits in the binary representation of the value.

        This operation is typically used with integer data types and has the involution property,
        meaning that applying it twice returns the original value: bitwise_not(bitwise_not(x)) = x.

        .. code-block:: mlir

            // Bitwise NOT with integer tensors
            %result = ttir.bitwise_not(%input, %output) : tensor<2x2xi32>, tensor<2x2xi32> -> tensor<2x2xi32>
            // Input tensor:
            // [[1, 2],
            //  [3, 4]]
            // Output tensor:
            // [[-2, -3],
            //  [-4, -5]]

            // Example with 8-bit integers
            %result = ttir.bitwise_not(%input, %output) : tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
            // Input: [0, 5, 255] (binary: [00000000, 00000101, 11111111])
            // Output: [255, 250, 0] (binary: [11111111, 11111010, 00000000])

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
        """
        return self.eltwise_proxy(
            torch.bitwise_not, ttir.BitwiseNotOp, [in0], unit_attrs=unit_attrs
        )

    # TTIR top level generic ops
    # class TTIR_GenericElementwiseUnaryOp

    def neg(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.neg``.

        *Elementwise negate operation.*

        Computes the negation of each element in the input tensor.
        For each element, returns the negation of the value.

        Mathematical definition: neg(x) = -x

        .. code-block:: mlir

            // Compute negation of all elements
            %result = ttir.neg(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [1.7, 2.0, -0.3, 4.5]
            // Output tensor:
            // [-1.7, -2.0, 0.3, -4.5]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            A tensor containing the negation of each input element
        """
        return self.eltwise_proxy(torch.neg, ttir.NegOp, [in0], unit_attrs=unit_attrs)

    def exp(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.exp``.

        *Elementwise exponential operation.*

        Computes the exponential of each element in the input tensor.
        For each element x, returns e^x, where e is Euler's number (approximately 2.71828).

        .. code-block:: mlir

            // Compute exponential of all elements
            %result = ttir.exp(%input, %output) : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensor:
            // [0.0, 1.0, 2.0]
            // Output tensor:
            // [1.0, 2.71828, 7.38906]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            A tensor containing the exponential of each element in the input tensor
        """
        return self.eltwise_proxy(torch.exp, ttir.ExpOp, [in0], unit_attrs)

    def log1p(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.log1p``.

        *Elementwise natural logarithm of one plus input operation.*

        Computes the natural logarithm of one plus each element in the input tensor.
        For each element x, returns ln(1 + x). This operation is more accurate than
        computing log(1 + x) directly for x values close to zero, where catastrophic
        cancellation can occur in the addition.

        Mathematical definition: log1p(x) = ln(1 + x)

        This operation is defined for x > -1. For values less than or equal to -1,
        the behavior depends on the implementation.

        .. code-block:: mlir

            // Compute log1p of all elements
            %result = ttir.log1p(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [0.0, -0.999, 7.0, 6.38905621, 15.0]
            // Output tensor:
            // [0.0, -6.90776825, 2.07944155, 2.0, 2.77258873]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            A tensor containing the log1p values of the input tensor
        """
        return self.eltwise_proxy(torch.log1p, ttir.Log1pOp, [in0], unit_attrs)

    def add(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.add``.

        *Elementwise addition operation.*

        Performs elementwise addition between two tensors.
        For each pair of corresponding elements, adds the element in the second
        tensor to the element in the first tensor.

        Mathematical definition: add(x, y) = x + y

        .. code-block:: mlir

            // Add corresponding elements
            %result = ttir.add(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensors:
            // lhs: [3.5, 0.0, -1.2]
            // rhs: [1.5, 2.0, -3.2]
            // Output tensor:
            // [5.0, 2.0, -4.4]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            A tensor containing the elementwise sum of the inputs
        """
        return self.eltwise_proxy(
            torch.add,
            ttir.AddOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def multiply(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.multiply``.

        *Elementwise multiplication operation.*

        Performs elementwise multiplication between two tensors.
        For each pair of corresponding elements, multiplies the element in the first
        tensor by the element in the second tensor.

        Mathematical definition: multiply(x, y) = x * y

        .. code-block:: mlir

            // Multiply corresponding elements
            %result = ttir.multiply(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensors:
            // lhs: [3.5, 0.0, -1.2]
            // rhs: [1.5, 2.0, -3.2]
            // Output tensor:
            // [5.25, 0.0, 3.84]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            A tensor containing the elementwise product of the inputs
        """
        return self.eltwise_proxy(
            torch.multiply,
            ttir.MultiplyOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def subtract(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.subtract``.

        *Elementwise subtraction operation.*

        Performs elementwise subtraction between two tensors.
        For each pair of corresponding elements, subtracts the element in the second
        tensor from the element in the first tensor.

        Mathematical definition: subtract(x, y) = x - y

        .. code-block:: mlir

            // Subtract corresponding elements
            %result = ttir.subtract(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensors:
            // lhs: [3.5, 0.0, -1.2]
            // rhs: [1.5, 2.0, -3.2]
            // Output tensor:
            // [2.0, -2.0, 2.0]

        Parameters
        ----------
        in0 : Operand
            First input tensor (minuend)
        in1 : Operand
            Second input tensor (subtrahend)
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            A tensor containing the elementwise difference of the inputs
        """
        return self.eltwise_proxy(
            torch.sub,
            ttir.SubtractOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def div(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.div``.

        *Elementwise division operation.*

        Performs elementwise division between two tensors.
        For each pair of corresponding elements, divides the element in the first
        tensor by the element in the second tensor.

        Mathematical definition: div(x, y) = x / y

        Note: The data type of the output tensor matches the data type of
        the input tensors.

        .. code-block:: mlir

            // Divide corresponding elements
            %result = ttir.div(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensors:
            // lhs: [1.5, 2.0, -1.2]
            // rhs: [0.5, 3.0, 2.2]
            // Output tensor:
            // [3.0, 0.6666667, -0.54545455]

        Parameters
        ----------
        in0 : Operand
            First input tensor (dividend)
        in1 : Operand
            Second input tensor (divisor)
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            A tensor containing the elementwise quotient of the inputs
        """
        return self.eltwise_proxy(
            torch.div, ttir.DivOp, [in0, in1], unit_attrs=unit_attrs
        )

    def maximum(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.maximum``.

        *Elementwise maximum operation.*

        Computes the elementwise maximum between two tensors.
        For each pair of corresponding elements, selects the larger value.

        Mathematical definition: maximum(x, y) = max(x, y)

        .. code-block:: mlir

            // Compute maximum of corresponding elements
            %result = ttir.maximum(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensors:
            // lhs: [1.5, 2.0, -1.2]
            // rhs: [0.5, 3.0, 2.2]
            // Output tensor:
            // [1.5, 3.0, 2.2]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            A tensor containing the elementwise maximum of the inputs
        """
        return self.eltwise_proxy(
            torch.maximum, ttir.MaximumOp, [in0, in1], unit_attrs=unit_attrs
        )

    def quantize(
        self,
        in0: Operand,
        scale: float,
        zero_point: int,
        dtype: torch.dtype,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.quantize``.

        *Quantize floating-point tensor to integer tensor.*

        Converts a floating-point tensor into a quantized integer tensor using the specified
        scale and zero_point parameters. For each element in the input tensor, computes:
            output[i] = (input[i] / scale) + zero_point

        .. code-block:: mlir

            // Quantize float32 tensor to int8
            %result = ttir.quantize(%input, %output) {scale = 0.1 : f32, zero_point = 128 : i32} : tensor<2x2xf32>, tensor<2x2xi8> -> tensor<2x2xi8>
            // Input tensor:
            // [[1.5, -0.2],
            //  [0.0, 3.7]]
            // Output tensor:
            // [[143, 126],
            //  [128, 165]]

        Parameters
        ----------
        in0 : Operand
            Input floating-point tensor to be quantized
        scale : float
            Scale factor for quantization (each integer step represents this value)
        zero_point : int
            Integer value that represents 0.0 in the quantized space
        dtype : torch.dtype
            Target integer data type for quantization (e.g., torch.int8)
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            The quantized integer tensor
        """
        golden_kwargs = {"scale": scale, "zero_point": zero_point, "dtype": dtype}
        return self.op_proxy(
            lambda *args, **kwargs: torch.quantize_per_tensor(
                *args, **kwargs
            ).int_repr(),
            ttir.QuantizeOp,
            [in0],
            golden_kwargs=golden_kwargs,
            output_type=self.get_type_from_torch_dtype(
                TypeInfo(dtype=dtype, scale=scale, zero_point=zero_point)
            ),
            unit_attrs=unit_attrs,
        )

    def dequantize(
        self,
        in0: Operand,
        scale: float,
        zero_point: int,
        dtype: torch.dtype,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.dequantize``.

        *Dequantize integer tensor to floating-point tensor.*

        Converts a quantized integer tensor back into a floating-point tensor using the
        specified scale and zero_point parameters. For each element in the input tensor,
        computes:
            output[i] = (input[i] - zero_point) * scale

        .. code-block:: mlir

            // Dequantize int8 tensor to float32
            %result = ttir.dequantize(%input, %output) {scale = 0.1 : f32, zero_point = 128 : i32} : tensor<2x2xi8>, tensor<2x2xf32> -> tensor<2x2xf32>
            // Input tensor:
            // [[143, 126],
            //  [128, 165]]
            // Output tensor:
            // [[1.5, -0.2],
            //  [0.0, 3.7]]

        Parameters
        ----------
        in0 : Operand
            Input quantized integer tensor to be dequantized
        scale : float
            Scale factor used in the original quantization
        zero_point : int
            Integer value that represents 0.0 in the quantized space
        dtype : torch.dtype
            Target floating-point data type (e.g., torch.float32)
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            The dequantized floating-point tensor
        """
        return self.op_proxy(
            torch.dequantize,
            ttir.DequantizeOp,
            [in0],
            output_type=self.get_type_from_torch_dtype(dtype=dtype),
            unit_attrs=unit_attrs,
        )

    def requantize(
        self,
        in0: Operand,
        scale: float,
        zero_point: int,
        dtype: torch.dtype,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.requantize``.

        *Requantize integer tensor to new scale and zero-point.*

        Converts a quantized integer tensor from one quantization scheme to another using
        new scale and zero-point parameters. For each element in the input tensor, computes:
            output[i] = round((input[i] - input_zero_point) * (input_scale / output_scale)) + output_zero_point

        .. code-block:: mlir

            // Requantize int8 tensor to new scale and zero-point
            %result = ttir.requantize(%input, %output) {scale = 0.2 : f32, zero_point = 100 : i32} : tensor<2x2xi8>, tensor<2x2xi8> -> tensor<2x2xi8>
            // Input tensor (scale=0.1, zero_point=128):
            // [[143, 126],
            //  [128, 165]]
            // Output tensor (scale=0.2, zero_point=100):
            // [[107, 98],
            //  [100, 119]]

        Parameters
        ----------
        in0 : Operand
            Input quantized integer tensor to be requantized
        scale : float
            New scale factor for requantization
        zero_point : int
            New integer value that represents 0.0 in the quantized space
        dtype : torch.dtype
            Target integer data type (e.g., torch.int8)
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            The requantized integer tensor with new scale and zero-point
        """
        golden_kwargs = {"scale": scale, "zero_point": zero_point, "dtype": dtype}
        return self.op_proxy(
            lambda *args, **kwargs: torch.quantize_per_tensor(
                torch.dequantize(args[0]), **kwargs
            ),
            ttir.RequantizeOp,
            [in0],
            golden_kwargs=golden_kwargs,
            output_type=self.get_type_from_torch_dtype(
                TypeInfo(dtype=dtype, scale=scale, zero_point=zero_point)
            ),
            unit_attrs=unit_attrs,
        )

    def to_layout(
        self,
        in0: Operand,
        output_type: RankedTensorType,
        unit_attrs: Optional[List[str]] = None,
        **kwargs,
    ) -> OpView:
        """
        Creates ``ttir.to_layout``.

        *Transform tensor to a different memory layout.*

        Transitions a tensor from one layout to another. This operation can handle:
        - Memory space transitions (e.g., DRAM to L1)
        - Data type conversions (e.g., f32 to f16)
        - Tile size changes (e.g., 1x16 to 32x32)
        - Tensor sharding modifications
        - Combinations of the above transformations

        .. code-block:: mlir

            // Transform tensor from system memory to L1 memory
            %result = ttir.to_layout(%input) : tensor<64x128xf32, #system> -> tensor<64x128xf32, #l1>
            // Input tensor in system memory:
            // [[1.5, 2.0, ...],
            //  [3.0, 4.0, ...]]
            // Output tensor in L1 memory (same values, different layout):
            // [[1.5, 2.0, ...],
            //  [3.0, 4.0, ...]]

        Parameters
        ----------
        in0 : Operand
            Input tensor to be transformed
        output_type : RankedTensorType
            Target type specifying the desired layout
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes
        **kwargs : dict
            Additional keyword arguments for layout transformation

        Returns
        -------
        *OpView*
            The tensor with transformed layout
        """
        return self.op_proxy(
            lambda *args, **kwargs: args[0],
            ttir.ToLayoutOp,
            [in0],
            output_type=output_type,
            output_create_fn=self.empty_from_tensor_type,
            organize_ttir_args=lambda i, o, _: (
                [self._get_type(o)],
                i[0],
                o,
            ),
            unit_attrs=unit_attrs,
            **kwargs,
        )

    def view_layout(
        self,
        in0: Operand,
        output_type: RankedTensorType,
        reinterpret_layout: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.view_layout``.

        *Create a new view of tensor with different layout.*

        Creates a new view of the input tensor with a different layout without copying
        or moving data. This is useful for reinterpreting the same data with different
        layout metadata.

        .. code-block:: mlir

            // Create a new view of tensor with different layout
            %result = ttir.view_layout(%input) {reinterpret_layout = false} : tensor<64x128xf32, #layout1> -> tensor<64x128xf32, #layout2>
            // Input tensor with layout1:
            // [[1.5, 2.0, ...],
            //  [3.0, 4.0, ...]]
            // Output tensor with layout2 (same data, different metadata):
            // [[1.5, 2.0, ...],
            //  [3.0, 4.0, ...]]

        Parameters
        ----------
        in0 : Operand
            Input tensor to create view from
        output_type : RankedTensorType
            Target type specifying the desired layout view
        reinterpret_layout : bool, optional
            If True, reinterprets the layout without validation
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            A new view of the tensor with the specified layout
        """
        return self.op_proxy(
            lambda *args, **kwargs: args[0],
            ttir.ViewLayoutOp,
            [in0],
            ttir_kwargs={"reinterpretLayout": reinterpret_layout},
            output_type=output_type,
            output_create_fn=self.empty_from_tensor_type,
            organize_ttir_args=lambda i, o, _: (
                self._get_type(o),
                i[0],
            ),
            unit_attrs=unit_attrs,
        )

    def tilize(
        self,
        in0: Operand,
        output_type: RankedTensorType,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.tilize``.

        *Convert tensor to tiled layout.*

        Transforms a tensor into a tiled layout format, where data is organized into
        regular blocks or tiles. This can improve memory access patterns and cache
        utilization for certain operations.

        .. code-block:: mlir

            // Convert tensor to tiled layout
            %result = ttir.tilize(%input) : tensor<128x128xf32> -> tensor<128x128xf32, #tiled<32x32>>
            // Input tensor (standard layout):
            // [[1.5, 2.0, ...],
            //  [3.0, 4.0, ...]]
            // Output tensor (tiled 32x32 layout):
            // Same values but organized in 32x32 tiles

        Parameters
        ----------
        in0 : Operand
            Input tensor to be tiled
        output_type : RankedTensorType
            Target type specifying the desired tiled layout
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            The tensor with tiled layout
        """
        return self.op_proxy(
            self.tilize_golden,
            ttir.ToLayoutOp,
            [in0],
            output_type=output_type,
            output_create_fn=self.empty_from_tensor_type,
            organize_ttir_args=lambda i, o, _: (
                [self._get_type(o)],
                i[0],
                o,
            ),
            unit_attrs=unit_attrs,
        )

    def untilize(
        self,
        in0: Operand,
        output_type: RankedTensorType,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.untilize``.

        *Convert tensor from tiled to standard layout.*

        Transforms a tensor from a tiled layout back to a standard row-major or
        column-major layout. This is the inverse operation of tilize.

        .. code-block:: mlir

            // Convert tensor from tiled to standard layout
            %result = ttir.untilize(%input) : tensor<128x128xf32, #tiled<32x32>> -> tensor<128x128xf32>
            // Input tensor (tiled 32x32 layout):
            // Data organized in 32x32 tiles
            // Output tensor (standard layout):
            // [[1.5, 2.0, ...],
            //  [3.0, 4.0, ...]]

        Parameters
        ----------
        in0 : Operand
            Input tensor with tiled layout
        output_type : RankedTensorType
            Target type specifying the desired standard layout
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        *OpView*
            The tensor with standard layout
        """
        return self.op_proxy(
            self.untilize_golden,
            ttir.ToLayoutOp,
            [in0],
            output_type=output_type,
            output_create_fn=self.empty_from_tensor_type,
            organize_ttir_args=lambda i, o, _: (
                [self._get_type(o)],
                i[0],
                o,
            ),
            unit_attrs=unit_attrs,
        )

    # CCL ops
    def mesh_shard(
        self,
        input: Operand,
        shard_type: str,
        shard_direction: str,
        shard_shape: Tuple[int, ...],
        shard_dims: Tuple[int, ...],
    ) -> OpView:
        """
        Creates ``ttir.mesh_shard``.

        *Shard a tensor across a device mesh.*

        Distributes a tensor across multiple devices in a mesh according to the specified
        sharding configuration. The sharding can be performed along one or more dimensions
        of the tensor.

        .. code-block:: mlir

            // Shard a tensor across a 2x2 device mesh
            %result = ttir.mesh_shard(%input) {
                shard_type = "block",
                shard_direction = "row",
                shard_shape = [2, 2],
                shard_dims = [0, 1]
            } : tensor<128x128xf32> -> tensor<64x64xf32>
            // Input tensor on single device:
            // [[1.0, 2.0, ...],
            //  [3.0, 4.0, ...]]
            // Output tensor sharded across devices:
            // Device 0: [[1.0, 2.0], [3.0, 4.0]]
            // Device 1: [[1.1, 2.1], [3.1, 4.1]]
            // Device 2: [[1.2, 2.2], [3.2, 4.2]]
            // Device 3: [[1.3, 2.3], [3.3, 4.3]]

        Parameters
        ----------
        input : Operand
            Input tensor to be sharded
        shard_type : str
            Type of sharding (e.g., "block", "cyclic")
        shard_direction : str
            Direction of sharding (e.g., "row", "col")
        shard_shape : Tuple[int, ...]
            Shape of the device mesh
        shard_dims : Tuple[int, ...]
            Tensor dimensions to shard along

        Returns
        -------
        *OpView*
        """
        kwargs = {
            "shard_type": Attribute.parse(shard_type),
            "shard_direction": Attribute.parse(shard_direction),
            "shard_shape": shard_shape,
            "shard_dims": shard_dims,
        }
        return self.ccl_proxy(
            mesh_shard_golden,
            ttir.MeshShardOp,
            [input],
            kwargs=kwargs,
        )

    def all_gather(
        self,
        input: Operand,
        all_gather_dim: int = None,
        cluster_axis: int = None,
    ) -> OpView:
        """
        Creates ``ttir.all_gather``.

        *Gather tensor data from all devices.*

        Collects tensor data from all devices in the system and concatenates them along
        the specified dimension. The gather operation can be performed along different
        axes of the device mesh.

        For a mesh shape of [2,4] with device IDs:
        [[0, 1, 2, 3],
         [4, 5, 6, 7]]

        - If cluster_axis=0: Gathers along columns (0,4), (1,5), (2,6), (3,7)
        - If cluster_axis=1: Gathers along rows (0,1,2,3), (4,5,6,7)

        .. code-block:: mlir

            // Gather tensor data from all devices along dimension 0
            %result = ttir.all_gather(%input) {all_gather_dim = 0, cluster_axis = 1} : tensor<32x64xf32> -> tensor<128x64xf32>
            // Input tensor on device 0:
            // [[1.0, 2.0],
            //  [3.0, 4.0]]
            // Output tensor after gathering:
            // [[1.0, 2.0],  // from device 0
            //  [5.0, 6.0],  // from device 1
            //  [9.0, 10.0], // from device 2
            //  [13.0, 14.0]] // from device 3

        Parameters
        ----------
        input : Operand
            Input tensor to be gathered
        all_gather_dim : int, optional
            Dimension along which to concatenate gathered tensors
        cluster_axis : int, optional
            Axis of device mesh for gathering (0 or 1)

        Returns
        -------
        *OpView*
        """
        kwargs = {"all_gather_dim": all_gather_dim, "cluster_axis": cluster_axis}
        return self.ccl_proxy(
            all_gather_golden,
            ttir.AllGatherOp,
            [input],
            kwargs=kwargs,
        )

    def all_reduce(
        self,
        input: Operand,
        reduce_type: str,
        cluster_axis: int,
    ) -> OpView:
        """
        Creates ``ttir.all_reduce``.

        *Reduce tensor data across all devices.*

        Performs a reduction operation (e.g., sum, max) across all devices in the system
        and broadcasts the result back to all devices. The reduction can be performed
        along different axes of the device mesh.

        For a mesh shape of [2,4] with device IDs:
        [[0, 1, 2, 3],
         [4, 5, 6, 7]]

        - If cluster_axis=0: Reduces along columns (0,4), (1,5), (2,6), (3,7)
        - If cluster_axis=1: Reduces along rows (0,1,2,3), (4,5,6,7)

        .. code-block:: mlir

            // Sum tensor data across all devices
            %result = ttir.all_reduce(%input) {reduce_type = "sum", cluster_axis = 1} : tensor<32x64xf32> -> tensor<32x64xf32>
            // Input tensor on device 0:
            // [[1.0, 2.0],
            //  [3.0, 4.0]]
            // Output tensor after reduction (same on all devices):
            // [[10.0, 20.0],  // sum of values from all devices
            //  [30.0, 40.0]]

        Parameters
        ----------
        input : Operand
            Input tensor to be reduced
        reduce_type : str
            Type of reduction operation (e.g., "sum", "max")
        cluster_axis : int
            Axis of device mesh for reduction (0 or 1)

        Returns
        -------
        *OpView*
        """
        kwargs = {
            "reduce_type": Attribute.parse(reduce_type),
            "cluster_axis": cluster_axis,
        }
        return self.ccl_proxy(
            all_reduce_golden,
            ttir.AllReduceOp,
            [input],
            kwargs=kwargs,
        )

    def reduce_scatter(
        self,
        input: Operand,
        reduce_type: str,
        scatter_dim: int,
        cluster_axis: int,
    ) -> OpView:
        """
        Creates ``ttir.reduce_scatter``.

        *Reduce tensor data and scatter results across devices.*

        Performs a reduction operation across all devices and then scatters different
        parts of the result to different devices. The reduction and scatter operations
        can be performed along different axes of the device mesh.

        For a mesh shape of [2,4] with device IDs:
        [[0, 1, 2, 3],
         [4, 5, 6, 7]]

        - If cluster_axis=0: Reduces along columns (0,4), (1,5), (2,6), (3,7)
        - If cluster_axis=1: Reduces along rows (0,1,2,3), (4,5,6,7)

        .. code-block:: mlir

            // Sum tensor data and scatter along dimension 0
            %result = ttir.reduce_scatter(%input) {reduce_type = "sum", scatter_dim = 0, cluster_axis = 1} : tensor<128x64xf32> -> tensor<32x64xf32>
            // Input tensor on each device:
            // [[1.0, 2.0],
            //  [3.0, 4.0]]
            // Output tensors after reduction and scatter:
            // Device 0: [[10.0, 20.0]]  // sum of first quarter
            // Device 1: [[30.0, 40.0]]  // sum of second quarter
            // Device 2: [[50.0, 60.0]]  // sum of third quarter
            // Device 3: [[70.0, 80.0]]  // sum of fourth quarter

        Parameters
        ----------
        input : Operand
            Input tensor to be reduced and scattered
        reduce_type : str
            Type of reduction operation (e.g., "sum", "max")
        scatter_dim : int
            Dimension along which to scatter the reduced results
        cluster_axis : int
            Axis of device mesh for reduction (0 or 1)

        Returns
        -------
        *OpView*
        """
        kwargs = {
            "reduce_type": Attribute.parse(reduce_type),
            "scatter_dim": scatter_dim,
            "cluster_axis": cluster_axis,
        }
        return self.ccl_proxy(
            reduce_scatter_golden,
            ttir.ReduceScatterOp,
            [input],
            kwargs=kwargs,
        )

    def collective_permute(
        self,
        input: Operand,
        source_target_pairs: List[Tuple[int, int]],
    ) -> OpView:
        """
        Creates ``ttir.collective_permute``.

        *Multi-device tensor permutation operation.*

        This operation ingests a multi-device tensor spread across devices and shuffles
        the data according to source_target_pairs [['src', 'dest']].

        .. code-block:: mlir

            // Example with 1x2 mesh - swap shards between devices 0 and 1
            %result = ttir.collective_permute(%input) {
              source_target_pairs = [[0, 1], [1, 0]]
            } : tensor<2x4xf32> -> tensor<2x4xf32>

            // Example with missing destination - device 0 shard becomes zeros
            %result = ttir.collective_permute(%input) {
              source_target_pairs = [[0, 1]]
            } : tensor<2x4xf32> -> tensor<2x4xf32>

        Parameters
        ----------
        input : Operand
            The input tensor to be permuted
        source_target_pairs : List[Tuple[int, int]]
            List of pairs of source and target device ids

        Returns
        -------
        *OpView*
        """
        kwargs = {
            "source_target_pairs": source_target_pairs,
        }
        return self.ccl_proxy(
            collective_permute_golden,
            ttir.CollectivePermuteOp,
            [input],
            kwargs=kwargs,
        )
