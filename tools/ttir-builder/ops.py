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
        """Get the size of the specified dimension of a tensor.

        Produces the size of the given dimension of the input tensor.

        Args:
            in0: Input tensor operand to get dimension size from
            dimension: The dimension index to get size of (default: 0)
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the size of the specified dimension

        Example:
            For input tensor of shape [3, 2, 7]:
            get_dimension_size(tensor, dimension=0) -> [3]
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
        """Perform a generalized dot product operation between two tensors.

        A flexible tensor operation that generalizes matrix multiplication by allowing user to specify which
        dimensions of two tensors to contract. Matrix multiplication is a special case of this operation,
        where the contraction happens along the last axis of the first tensor and the second-to-last axis
        of the second tensor.

        Based on StableHLO DotGeneral Op (https://openxla.org/stablehlo/spec#dot_general)

        Args:
            in0: Left-hand side input tensor
            in1: Right-hand side input tensor
            out0: Output tensor
            batch_dims_lhs: Batch dimensions for the left-hand side tensor
            contract_dims_lhs: Contracting dimensions for the left-hand side tensor
            batch_dims_rhs: Batch dimensions for the right-hand side tensor
            contract_dims_rhs: Contracting dimensions for the right-hand side tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The result of the dot general operation
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
        """Elementwise conditional selection operation based on a predicate.

        For each element position, it selects between two values based on a boolean condition:
        - If the condition is true (non-zero), it selects the corresponding element from the second tensor
        - If the condition is false (zero), it selects the corresponding element from the third tensor

        This operation supports broadcasting, allowing inputs of different shapes to be combined
        according to standard broadcasting rules. It is equivalent to the ternary conditional operator
        (condition ? true_value : false_value) applied elementwise across tensors.

        Args:
            in0: Condition tensor (predicate)
            in1: Tensor containing values to select when condition is true
            in2: Tensor containing values to select when condition is false
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing elements selected from in1 where condition is true,
                   and from in2 where condition is false

        Example:
            # Select elements from true_values where condition is true,
            # otherwise select from false_values
            result = where(condition, true_values, false_values)

            # With broadcasting (condition is a scalar)
            result = where(scalar_condition, true_values, false_values)
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

    def abs(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise absolute value operation.

        The abs operation computes the absolute value of each element in the input tensor.
        For each element, it returns the magnitude of the value without regard to its sign:
        - For real numbers, it returns |x| (the non-negative value without sign)

        This operation has the idempotence property, meaning that applying it multiple times
        produces the same result as applying it once: abs(abs(x)) = abs(x). The operation
        preserves the data type of the input.

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the absolute values of the input tensor

        Example:
            # Compute absolute values of all elements in input
            # Input tensor: [[-2.5, 3.7, 0.0, 1.2], ...]
            result = abs(input)
        """
        return self.eltwise_proxy(torch.abs, ttir.AbsOp, [in0], unit_attrs)

    def cbrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise cubic root operation.

        The `cbrt` operation computes the cubic root (∛) of each element in the input tensor.
        For each element, it returns the real-valued number that, when cubed, equals the input value.
        Unlike square root, cubic root is defined for negative numbers as well as positive numbers.

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the cubic root of each element in the input tensor

        Example:
            # Compute cubic root of all elements in input
            # Input tensor: [[8.0, 27.0, -8.0, 1.0], ...]
            # Output tensor: [[2.0, 3.0, -2.0, 1.0], ...]
            result = cbrt(input)
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

    def ceil(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise ceiling operation.

        The `ceil` operation computes the ceiling (smallest integer greater than or equal to x)
        of each element in the input tensor. For each element, it rounds the value up to the
        nearest integer. The operation preserves the data type of the input.

        This operation has the idempotence property, meaning that applying it multiple times
        produces the same result as applying it once: ceil(ceil(x)) = ceil(x).

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the ceiling values of the input tensor

        Example:
            # Compute ceiling of all elements in input
            # Input tensor: [[1.7, 2.0, -0.3, 4.5], ...]
            # Output tensor: [[2.0, 2.0, 0.0, 5.0], ...]
            result = ceil(input)
        """
        return self.eltwise_proxy(torch.ceil, ttir.CeilOp, [in0], unit_attrs)

    def cos(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise cosine operation.

        The cos operation computes the cosine of each element in the input tensor.
        For each element, it returns the cosine of the angle in radians.

        Args:
            in0: Input tensor containing angles in radians
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the cosine values of the input tensor

        Example:
            # Compute cosine of all elements in input
            # Input tensor: [[1.7, 2.0, -0.3, 4.5], ...]
            # Output tensor: [[0.9601, 0.5403, -0.9553, -0.1365], ...]
            result = cos(input)
        """
        return self.eltwise_proxy(torch.cos, ttir.CosOp, [in0], unit_attrs)

    def floor(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise floor operation.

        The `floor` operation computes the floor (greatest integer less than or equal to x)
        of each element in the input tensor. For each element, it rounds the value down to
        the nearest integer. The operation preserves the data type of the input.

        This operation has the idempotence property, meaning that applying it multiple times
        produces the same result as applying it once: floor(floor(x)) = floor(x).

        Mathematical definition: floor(x) = ⌊x⌋ = max{n ∈ ℤ | n ≤ x}

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the floor values of the input tensor

        Example:
            # Compute floor of all elements in input
            # Input tensor: [[1.7, 2.0, -0.3, 4.5], ...]
            # Output tensor: [[1.0, 2.0, -1.0, 4.0], ...]
            result = floor(input)
        """
        return self.eltwise_proxy(torch.floor, ttir.FloorOp, [in0], unit_attrs)

    def gelu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise GELU operation.

        The `gelu` operation computes the GELU (Gaussian Error Linear Unit) of each element
        in the input tensor. For each element, it returns the GELU value, which is a smooth,
        non-monotonic function that approximates the cumulative distribution function of a
        standard normal distribution.

        Mathematical definition: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the GELU values of the input tensor

        Example:
            # Compute GELU of all elements in input
            # Input tensor: [[1.7, 2.0, -0.3, 4.5], ...]
            # Output tensor: [[0.9601, 0.5403, -0.3, 4.5], ...]
            result = gelu(input)
        """
        return self.eltwise_proxy(
            torch.nn.functional.gelu, ttir.GeluOp, [in0], unit_attrs
        )

    def is_finite(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise isfinite operation.

        The isfinite operation checks if each element in the input tensor is finite
        (neither infinite nor NaN). For each element, it returns a boolean value
        indicating whether the element is finite.

        Mathematical definition: isfinite(x) = x ∈ ℝ

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing boolean values indicating whether each input element is finite

        Example:
            # Check if all elements in input are finite
            # Input tensor: [[1.7, 2.0, Inf, 4.5], ...]
            # Output tensor: [[true, true, false, true], ...]
            result = is_finite(input)
        """
        return self.eltwise_proxy(torch.isfinite, ttir.IsFiniteOp, [in0], unit_attrs)

    def logical_not(
        self, in0: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Elementwise logical not operation.

        The `logical_not` operation computes the logical negation of each element in the
        input tensor. For each element, it returns a boolean value indicating whether
        the element is false (zero) or true (non-zero).

        Mathematical definition: logical_not(x) = !x

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the logical negation of each input element

        Example:
            # Compute logical negation of all elements in input
            # Input tensor: [[1.7, 2.0, -0.0, 4.5], ...]
            # Output tensor: [[false, false, true, false], ...]
            result = logical_not(input)
        """
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.logical_not,
            ttir.LogicalNotOp,
            [in0],
            golden_kwargs={"out": golden_output},
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def bitwise_not(
        self, in0: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Elementwise bitwise NOT operation.

        The bitwise_not operation computes the bitwise NOT (one's complement) of each element
        in the input tensor. For each element, it flips all the bits in the binary representation
        of the value.

        This operation is typically used with integer data types and has the involution property,
        meaning that applying it twice returns the original value: bitwise_not(bitwise_not(x)) = x.

        Mathematical definition: bitwise_not(x) = ~x

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the bitwise NOT of each input element

        Example:
            # Bitwise NOT with integer tensors
            # Input tensor: [[1, 2], [3, 4]]
            # Output tensor: [[-2, -3], [-4, -5]]
            result = bitwise_not(input)

            # Example with 8-bit integers
            # Input: [0, 5, 255] (binary: [00000000, 00000101, 11111111])
            # Output: [255, 250, 0] (binary: [11111111, 11111010, 00000000])
        """
        return self.eltwise_proxy(
            torch.bitwise_not, ttir.BitwiseNotOp, [in0], unit_attrs
        )

    def neg(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise negate operation.

        The `neg` operation negates each element in the input tensor. For each element,
        it returns the negation of the value. The operation preserves the data type
        of the input.

        Mathematical definition: neg(x) = -x

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the negation of each input element

        Example:
            # Compute negation of all elements in input
            # Input tensor: [[1.7, 2.0, -0.3, 4.5], ...]
            # Output tensor: [[-1.7, -2.0, 0.3, -4.5], ...]
            result = neg(input)
        """
        return self.eltwise_proxy(torch.neg, ttir.NegOp, [in0], unit_attrs)

    # NOTE: See issue #1719 for information on golden PCC fail
    def tan(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise tan operation.

        The `tan` operation computes the tangent of each element in the input tensor.
        For each element, it returns the tangent of the angle in radians.

        Args:
            in0: Input tensor containing angles in radians
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the tangent values of the input tensor

        Example:
            # Compute tangent of all elements in input
            # Input tensor: [[1.7, 2.0, -0.3, 4.5], ...]
            # Output tensor: [[0.9601, 0.5403, -0.3, 4.5], ...]
            result = tan(input)
        """
        return self.eltwise_proxy(torch.tan, ttir.TanOp, [in0], unit_attrs)

    def atan(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise arctangent operation.

        The `atan` operation computes the arctangent (inverse tangent) of each element in
        the input tensor. For each element, it returns the angle in radians whose tangent
        is the input value. The operation returns values in the range [-π/2, π/2].

        Mathematical definition: atan(x) = tan⁻¹(x), where the result is in the range [-π/2, π/2]

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the arctangent values of the input tensor

        Example:
            # Compute arctangent of all elements in input
            # Input tensor: [1.0, 0.5, 0.0, -1.0]
            # Output tensor: [0.785, 0.464, 0.0, -0.785]  # values in radians

            # Example with different values
            # Input tensor: [0.0, 1.0, 1000.0]
            # Output tensor: [0.0, 0.785, 1.571]  # values approach π/2 as input grows
            result = atan(input)
        """
        return self.eltwise_proxy(torch.atan, ttir.AtanOp, [in0], unit_attrs)

    def tanh(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise hyperbolic tangent operation.

        The `tanh` operation computes the hyperbolic tangent of each element in the input
        tensor. For each element, it returns the hyperbolic tangent of the value.

        Mathematical definition: tanh(x) = (e^x - e^-x) / (e^x + e^-x)

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the hyperbolic tangent values of the input tensor

        Example:
            # Compute hyperbolic tangent of all elements in input
            # Input tensor: [[1.7, 2.0, -0.3, 4.5], ...]
            # Output tensor: [[0.9601, 0.5403, -0.3, 4.5], ...]
            result = tanh(input)
        """
        return self.eltwise_proxy(torch.tanh, ttir.TanhOp, [in0], unit_attrs)

    def reciprocal(
        self, in0: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Elementwise reciprocal operation.

        The `reciprocal` operation computes the reciprocal (1/x) of each element in the
        input tensor. For each element, it returns the reciprocal of the value.

        Mathematical definition: reciprocal(x) = 1 / x

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the reciprocal values of the input tensor

        Example:
            # Compute reciprocal of all elements in input
            # Input tensor: [[1.7, 2.0, -0.3, 4.5], ...]
            # Output tensor: [[0.5882, 0.5, -3.3333, 0.2173], ...]
            result = reciprocal(input)
        """
        return self.eltwise_proxy(
            torch.reciprocal, ttir.ReciprocalOp, [in0], unit_attrs
        )

    def relu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise ReLU operation.

        The `relu` operation computes the rectified linear unit (ReLU) of each element in
        the input tensor. For each element, it returns the maximum of 0 and the value.
        The operation preserves the data type of the input.

        Mathematical definition: relu(x) = max(0, x)

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the ReLU values of the input tensor

        Example:
            # Compute ReLU of all elements in input
            # Input tensor: [[1.7, 2.0, -0.3, 4.5], ...]
            # Output tensor: [[1.7, 2.0, 0.0, 4.5], ...]
            result = relu(input)
        """
        return self.eltwise_proxy(torch.relu, ttir.ReluOp, [in0], unit_attrs)

    def rsqrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise reciprocal square root operation.

        The rsqrt operation computes the reciprocal square root of each element in the
        input tensor. For each element, it returns the reciprocal of the square root
        of the value.

        Mathematical definition: rsqrt(x) = 1 / sqrt(x)

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the reciprocal square root values of the input tensor

        Example:
            # Compute reciprocal square root of all elements in input
            # Input tensor: [[1.7, 2.0, -0.3, 4.5], ...]
            # Output tensor: [[0.5882, 0.5, -3.3333, 0.2173], ...]
            result = rsqrt(input)
        """
        return self.eltwise_proxy(torch.rsqrt, ttir.RsqrtOp, [in0], unit_attrs)

    def sigmoid(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise sigmoid operation.

        The sigmoid operation computes the sigmoid of each element in the input tensor.
        For each element, it returns the sigmoid of the value.

        Mathematical definition: sigmoid(x) = 1 / (1 + exp(-x))

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the sigmoid values of the input tensor

        Example:
            # Compute sigmoid of all elements in input
            # Input tensor: [[1.7, 2.0, -0.3, 4.5], ...]
            # Output tensor: [[0.8391, 0.9641, 0.5793, 0.9899], ...]
            result = sigmoid(input)
        """
        return self.eltwise_proxy(torch.sigmoid, ttir.SigmoidOp, [in0], unit_attrs)

    def sign(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise sign operation.

        The sign operation computes the sign of each element in the input tensor.
        For each element, it returns:
        - 1 if the value is positive
        - 0 if the value is zero
        - -1 if the value is negative

        This operation has the idempotence property, meaning that applying it multiple times
        produces the same result as applying it once: sign(sign(x)) = sign(x).

        Mathematical definition: sign(x) = {
            1  if x > 0
            0  if x = 0
            -1 if x < 0
        }

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the sign values of the input tensor

        Example:
            # Compute sign of all elements in input
            # Input tensor:
            # [[3, -2, 0],
            #  [1, -4, 4]]
            # Output tensor:
            # [[1, -1, 0],
            #  [1, -1, 1]]

            # Example with floating-point values
            # Input tensor: [5.7, -0.0, 0.001, -3.14]
            # Output tensor: [1.0, 0.0, 1.0, -1.0]
            result = sign(input)
        """
        return self.eltwise_proxy(torch.sign, ttir.SignOp, [in0], unit_attrs)

    def sin(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise sine operation.

        The sin operation computes the sine of each element in the input tensor.
        For each element, it returns the sine of the angle in radians.

        Args:
            in0: Input tensor containing angles in radians
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the sine values of the input tensor

        Example:
            # Compute sine of all elements in input
            # Input tensor: [[1.7, 2.0, -0.3, 4.5], ...]
            # Output tensor: [[0.9601, 0.5403, -0.3, 4.5], ...]
            result = sin(input)
        """
        return self.eltwise_proxy(torch.sin, ttir.SinOp, [in0], unit_attrs)

    def sqrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise square root operation.

        The sqrt operation computes the square root of each element in the input tensor.
        For each element, it returns the square root of the value.

        Mathematical definition: sqrt(x) = √x

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the square root values of the input tensor

        Example:
            # Compute square root of all elements in input
            # Input tensor: [[1.7, 2.0, -0.3, 4.5], ...]
            # Output tensor: [[0.5882, 0.5, -3.3333, 0.2173], ...]
            result = sqrt(input)
        """
        return self.eltwise_proxy(torch.sqrt, ttir.SqrtOp, [in0], unit_attrs)

    def typecast(
        self, in0: Operand, out: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Elementwise type casting operation.

        The typecast operation converts each element in the input tensor to a different data
        type. This operation performs element-wise type conversion, such as converting from
        integers to floating-point values or between different floating-point precisions.
        The conversion follows the standard type conversion rules for the target platform.

        Args:
            in0: Input tensor
            out: Output tensor with the target data type
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the type-converted values of the input tensor

        Example:
            # Cast from int32 to float32
            # Input tensor: [[1, 2, -3, 4], ...]
            # Output tensor: [[1.0, 2.0, -3.0, 4.0], ...]
            result = typecast(input, float_output)

            # Cast from float32 to int32 (truncation, not rounding)
            # Input tensor: [1.7, -2.3, 3.0]
            # Output tensor: [1, -2, 3]
            result = typecast(float_input, int_output)

            # Cast from float32 to float64 (higher precision)
            # Input tensor: [3.14159, 2.71828]
            # Output tensor: [3.14159, 2.71828]  # Same values but with higher precision
            result = typecast(f32_input, f64_output)
        """
        output_type = self.get_type_from_torch_dtype(self._get_golden_tensor(out).dtype)
        return self.op_proxy(
            torch.Tensor.type,
            ttir.TypecastOp,
            [in0],
            golden_kwargs={"dtype": self._get_golden_tensor(out).type()},
            output_type=output_type,
            unit_attrs=unit_attrs,
        )

    def log(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise natural logarithm operation.

        The `log` operation computes the natural logarithm of each element in the input tensor.
        For each element, it returns the natural logarithm (base e) of the value.

        This operation is defined only for positive values; the behavior for zero or negative
        inputs depends on the implementation (may return NaN, infinity, or other special values).

        Mathematical definition: log(x) = ln(x), where ln is the natural logarithm

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the natural logarithm values of the input tensor

        Example:
            # Compute natural logarithm of all elements in input
            # Input tensor: [[1.0, 2.718, 7.389, 20.086], ...]
            # Output tensor: [[0.0, 1.0, 2.0, 3.0], ...]

            # Example with different values
            # Input tensor: [10.0, 100.0, 1000.0]
            # Output tensor: [2.303, 4.605, 6.908]  # ln(10), ln(100), ln(1000)
            result = log(input)
        """
        return self.eltwise_proxy(torch.log, ttir.LogOp, [in0], unit_attrs)

    def log1p(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise natural logarithm of one plus input operation.

        The `log1p` operation computes the natural logarithm of one plus each element in the
        input tensor. For each element x, it returns ln(1 + x). This operation is more
        accurate than computing log(1 + x) directly for x values close to zero, and it is
        defined for x > -1. For values less than or equal to -1, the behavior depends on
        the implementation (may return NaN or negative infinity).

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the log1p values of the input tensor

        Example:
            # Compute log1p of all elements in input
            # Input tensor: [0.0, -0.999, 7.0, 6.38905621, 15.0]
            # Output tensor: [0.0, -6.90776825, 2.07944155, 2.0, 2.77258873]

            # Example with small values where log1p is more accurate than log(1+x)
            # Input tensor: [1e-10, 1e-7, 1e-5]
            # Output tensor: [1e-10, 1e-7, 1e-5]  # Approximately equal to input for small values
            result = log1p(input)
        """
        return self.eltwise_proxy(torch.log1p, ttir.Log1pOp, [in0], unit_attrs)

    def expm1(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise exponential minus one operation.

        The `expm1` operation computes the exponential of each element in the input tensor
        and subtracts one. For each element x, it returns e^x - 1. This operation is more
        accurate than computing exp(x) - 1 directly for x values close to zero, where
        catastrophic cancellation can occur in the subtraction.

        Args:
            in0: Input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the expm1 values of the input tensor

        Example:
            # Compute expm1 of all elements in input
            # Input tensor:
            # [[0.0, 1.0],
            #  [0.0, 0.0]]
            # Output tensor:
            # [[0.0, 1.71828],
            #  [0.0, 0.0]]

            # Example with small values where expm1 is more accurate than exp(x)-1
            # Input tensor: [1e-10, 1e-7, 1e-5]
            # Output tensor: [1e-10, 1e-7, 1e-5]  # Approximately equal to input for small values
            result = expm1(input)
        """
        return self.eltwise_proxy(torch.expm1, ttir.Expm1Op, [in0], unit_attrs)

    # class TTIR_ElementwiseUnaryWithFloatParameterOp

    def leaky_relu(
        self,
        in0: Operand,
        parameter: float = 0.01,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Elementwise leaky ReLU operation.

        The `leaky_relu` operation computes an element-wise
        activation function over its input tensor. It is defined as:

        y = x if x > 0
        y = parameter * x if x <= 0

        where `parameter` is a small, user-defined constant that determines the slope for
        negative inputs.

        Args:
            in0: Input tensor to be activated
            parameter: The slope for negative values (default: 0.01)
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The tensor after applying the Leaky ReLU activation

        Example:
            # Compute Leaky ReLU with default parameter (0.01)
            # Input tensor: [[1.0, -2.0, 3.0, -4.0], ...]
            # Output tensor: [[1.0, -0.02, 3.0, -0.04], ...]
            result = leaky_relu(input)

            # Compute Leaky ReLU with custom parameter (0.1)
            # Input tensor: [[1.0, -2.0, 3.0, -4.0], ...]
            # Output tensor: [[1.0, -0.2, 3.0, -0.4], ...]
            result = leaky_relu(input, parameter=0.1)
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

    def eq(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Elementwise equality comparison operation.

        The `eq` operation performs an elementwise equality comparison between two tensors.
        For each pair of corresponding elements, it returns:
        - 1 (true) if the elements are equal
        - 0 (false) if the elements are not equal

        Note that special handling may be required for floating-point NaN values, as NaN is not
        equal to any value, including itself.

        Mathematical definition: equal(x, y) = x == y

        Args:
            in0: First input tensor
            in1: Second input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A boolean tensor with 1s where elements are equal and 0s where they differ

        Example:
            # Compare elements for equality
            # Input tensors:
            # lhs: [[1.0, 2.0, 3.0, 2.0], ...]
            # rhs: [[1.0, 2.0, 4.0, 5.0], ...]
            # Output tensor: [[1, 1, 0, 0], ...]  # 1 where equal, 0 where not equal
            result = eq(lhs, rhs)

            # Example with integer tensors
            # Input tensors:
            # int_lhs: [10, -5, 0]
            # int_rhs: [10, 5, 1]
            # Output tensor: [1, 0, 0]  # Only the first elements are equal
            result = eq(int_lhs, int_rhs)
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
        """Elementwise inequality comparison operation.

        The `ne` operation performs an elementwise inequality comparison between two tensors.
        For each pair of corresponding elements, it returns:
        - 1 (true) if the elements are not equal
        - 0 (false) if the elements are equal

        Note that special handling may be required for floating-point NaN values, as NaN is not
        equal to any value, including itself. This means ne(NaN, NaN) should return true.

        Mathematical definition: not_equal(x, y) = x != y

        Args:
            in0: First input tensor
            in1: Second input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A boolean tensor with 1s where elements differ and 0s where they are equal

        Example:
            # Compare elements for inequality
            # Input tensors:
            # lhs: [[1.0, 2.0, 3.0, 2.0], ...]
            # rhs: [[1.0, 2.0, 4.0, 5.0], ...]
            # Output tensor: [[0, 0, 1, 1], ...]  # 0 where equal, 1 where not equal
            result = ne(lhs, rhs)

            # Example with integer tensors
            # Input tensors:
            # int_lhs: [10, -5, 0]
            # int_rhs: [10, 5, 1]
            # Output tensor: [0, 1, 1]  # Only the first elements are equal, so their result is 0
            result = ne(int_lhs, int_rhs)
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
        """Elementwise greater than or equal to comparison operation.

        The `ge` operation performs an elementwise greater than or equal to comparison between
        two tensors. For each pair of corresponding elements, it returns:
        - 1 (true) if the left element is greater than or equal to the right element
        - 0 (false) if the left element is less than the right element

        Mathematical definition: greater_equal(x, y) = x >= y

        Args:
            in0: First input tensor (left-hand side)
            in1: Second input tensor (right-hand side)
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A boolean tensor with 1s where left >= right and 0s otherwise

        Example:
            # Compare elements for greater than or equal to
            # Input tensors:
            # lhs: [[1.0, 2.0, 3.0, 2.0], ...]
            # rhs: [[1.0, 2.0, 4.0, 5.0], ...]
            # Output tensor: [[1, 1, 0, 0], ...]  # 1 where greater or equal, 0 where less
            result = ge(lhs, rhs)

            # Example with integer tensors
            # Input tensors:
            # int_lhs: [10, -5, 0]
            # int_rhs: [10, 5, 1]
            # Output tensor: [1, 0, 0]  # Only the first elements are greater or equal
            result = ge(int_lhs, int_rhs)
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
        """Elementwise greater than comparison operation.

        The `gt` operation performs an elementwise greater than comparison between two tensors.
        For each pair of corresponding elements, it returns:
        - 1 (true) if the left element is greater than the right element
        - 0 (false) if the left element is less than or equal to the right element

        Mathematical definition: greater(x, y) = x > y

        Args:
            in0: First input tensor (left-hand side)
            in1: Second input tensor (right-hand side)
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A boolean tensor with 1s where left > right and 0s otherwise

        Example:
            # Compare elements for greater than
            # Input tensors:
            # lhs: [[1.0, 2.0, 3.0, 2.0], ...]
            # rhs: [[1.0, 1.0, 4.0, 5.0], ...]
            # Output tensor: [[0, 1, 0, 0], ...]  # 1 where greater, 0 where less or equal
            result = gt(lhs, rhs)

            # Example with integer tensors
            # Input tensors:
            # int_lhs: [10, -5, 0]
            # int_rhs: [9, 5, 0]
            # Output tensor: [1, 0, 0]  # Only the first element is greater
            result = gt(int_lhs, int_rhs)
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
        """Elementwise less than or equal to comparison operation.

        The `le` operation performs an elementwise less than or equal to comparison between
        two tensors. For each pair of corresponding elements, it returns:
        - 1 (true) if the left element is less than or equal to the right element
        - 0 (false) if the left element is greater than the right element

        Mathematical definition: less_equal(x, y) = x <= y

        Args:
            in0: First input tensor (left-hand side)
            in1: Second input tensor (right-hand side)
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A boolean tensor with 1s where left <= right and 0s otherwise

        Example:
            # Compare elements for less than or equal to
            # Input tensors:
            # lhs: [[1.0, 2.0, 3.0, 2.0], ...]
            # rhs: [[1.0, 2.0, 4.0, 5.0], ...]
            # Output tensor: [[1, 1, 1, 0], ...]  # 1 where less or equal, 0 where greater
            result = le(lhs, rhs)

            # Example with integer tensors
            # Input tensors:
            # int_lhs: [10, -5, 0]
            # int_rhs: [10, 5, 1]
            # Output tensor: [1, 1, 1]  # All elements are less or equal
            result = le(int_lhs, int_rhs)
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
        """Elementwise less than comparison operation.

        The `lt` operation performs an elementwise less than comparison between two tensors.
        For each pair of corresponding elements, it returns:
        - 1 (true) if the left element is less than the right element
        - 0 (false) if the left element is greater than or equal to the right element

        Mathematical definition: less(x, y) = x < y

        Args:
            in0: First input tensor (left-hand side)
            in1: Second input tensor (right-hand side)
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A boolean tensor with 1s where left < right and 0s otherwise

        Example:
            # Compare elements for less than
            # Input tensors:
            # lhs: [[1.0, 2.0, 3.0, 2.0], ...]
            # rhs: [[1.0, 2.0, 4.0, 5.0], ...]
            # Output tensor: [[0, 0, 1, 1], ...]  # 1 where less, 0 where greater or equal
            result = lt(lhs, rhs)

            # Example with integer tensors
            # Input tensors:
            # int_lhs: [10, -5, 0]
            # int_rhs: [10, 5, 1]
            # Output tensor: [0, 1, 1]  # Second and third elements are less
            result = lt(int_lhs, int_rhs)
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
        """Elementwise logical AND operation.

        The `logical_and` operation performs an elementwise logical AND operation between
        two tensors. For each pair of corresponding elements, it returns:
        - 1 (true) if both elements are 1 (true)
        - 0 (false) if at least one element is 0 (false)

        This operation is idempotent, meaning logical_and(x, x) = x.

        Args:
            in0: First input tensor
            in1: Second input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A boolean tensor with 1s where both inputs are true and 0s otherwise

        Example:
            # Logical AND operation
            # Input tensors:
            # lhs: [[1, 0, 1, 0], ...]
            # rhs: [[1, 1, 0, 1], ...]
            # Output tensor: [[1, 0, 0, 0], ...]  # 1 where both are 1, 0 otherwise
            result = logical_and(lhs, rhs)

            # Example with integer tensors
            # Input tensors:
            # int_lhs: [10, 0, 0]
            # int_rhs: [10, 5, 1]
            # Output tensor: [1, 0, 0]  # Only the first element is true
            result = logical_and(int_lhs, int_rhs)
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
        """Elementwise logical OR operation.

        The `logical_or` operation performs an elementwise logical OR operation between
        two tensors. For each pair of corresponding elements, it returns:
        - 1 (true) if at least one element is 1 (true)
        - 0 (false) if both elements are 0 (false)

        This operation is idempotent, meaning logical_or(x, x) = x.

        Mathematical definition: logical_or(x, y) = x || y

        Args:
            in0: First input tensor
            in1: Second input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A boolean tensor with 1s where at least one input is true and 0s otherwise

        Example:
            # Logical OR operation
            # Input tensors:
            # lhs: [[1, 0, 1, 0], ...]
            # rhs: [[1, 1, 0, 1], ...]
            # Output tensor: [[1, 1, 1, 1], ...]  # 1 where at least one is 1, 0 otherwise
            result = logical_or(lhs, rhs)

            # Example with integer tensors
            # Input tensors:
            # int_lhs: [10, 0, 0]
            # int_rhs: [10, 5, 1]
            # Output tensor: [1, 1, 1]  # All elements are true
            result = logical_or(int_lhs, int_rhs)
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
        return self.eltwise_proxy(
            torch.bitwise_and, ttir.BitwiseAndOp, [in0, in1], unit_attrs=unit_attrs
        )

    @autodoc_skip
    def bitwise_or(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        return self.eltwise_proxy(
            torch.bitwise_or, ttir.BitwiseOrOp, [in0, in1], unit_attrs=unit_attrs
        )

    @autodoc_skip
    def bitwise_xor(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Elementwise bitwise XOR operation.

        The bitwise_xor operation performs an elementwise bitwise XOR (exclusive OR)
        operation between two tensors. For each pair of corresponding elements, it
        computes the bitwise XOR of their binary representations. This operation is
        typically used with integer data types and has the property that when applied
        twice with the same second operand, it returns the original input:
        bitwise_xor(bitwise_xor(x, y), y) = x.

        Args:
            in0: First input tensor
            in1: Second input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the bitwise XOR of the inputs

        Example:
            # Bitwise XOR operation
            # Input tensors:
            # lhs: [[1, 2], [3, 4]]
            # rhs: [[5, 6], [7, 8]]
            # Output tensor: [[4, 4], [4, 12]]
            result = bitwise_xor(lhs, rhs)

            # Example with binary representation (for 8-bit integers)
            # Input tensors:
            # int8_lhs: [0x0F, 0xAA, 0xFF, 0x00]  (binary: [00001111, 10101010, 11111111, 00000000])
            # int8_rhs: [0xF0, 0x55, 0xFF, 0x00]  (binary: [11110000, 01010101, 11111111, 00000000])
            # Output tensor: [0xFF, 0xFF, 0x00, 0x00]  (binary: [11111111, 11111111, 00000000, 00000000])
            result = bitwise_xor(int8_lhs, int8_rhs)
        """
        return self.eltwise_proxy(
            torch.bitwise_xor, ttir.BitwiseXorOp, [in0, in1], unit_attrs=unit_attrs
        )

    def minimum(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Elementwise minimum operation.

        The `minimum` operation computes the elementwise minimum between two tensors.
        For each pair of corresponding elements, it selects the smaller value and places
        it in the output tensor. This operation has the idempotence property, meaning
        that applying it twice with the same second operand returns the original result:
        minimum(minimum(x, y), y) = minimum(x, y).

        Note: When comparing with NaN values, NaN is typically not selected as the minimum value.

        Args:
            in0: First input tensor
            in1: Second input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the elementwise minimum of the inputs

        Example:
            # Minimum operation
            # Input tensors:
            # lhs: [[3, 2, 7], [1, 4, 4]]
            # rhs: [[1, 4, 2], [1, 2, 3]]
            # Output tensor: [[1, 2, 2], [1, 2, 3]]
            result = minimum(lhs, rhs)

            # Example with floating point values
            # Input tensors:
            # float_lhs: [3.5, -2.1, 0.0]
            # float_rhs: [1.2, -5.0, 0.0]
            # Output tensor: [1.2, -5.0, 0.0]
            result = minimum(float_lhs, float_rhs)
        """
        return self.eltwise_proxy(
            torch.minimum, ttir.MinimumOp, [in0, in1], unit_attrs=unit_attrs
        )

    def subtract(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Elementwise subtract operation.

        The subtract operation performs an elementwise subtraction between two tensors.
        For each pair of corresponding elements, it subtracts the element in the second
        tensor from the element in the first tensor and places the result in the output
        tensor.

        Mathematical definition: subtract(x, y) = x - y

        Note: The data type of the output tensor matches the data type of the input tensors.

        Args:
            in0: First input tensor (minuend)
            in1: Second input tensor (subtrahend)
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the elementwise difference of the inputs

        Example:
            # Subtraction operation
            # Input tensors:
            # lhs: [10, 20, 30]
            # rhs: [1, 2, 3]
            # Output tensor: [9, 18, 27]
            result = subtract(lhs, rhs)

            # Example with floating point values
            # Input tensors:
            # float_lhs: [3.5, 0.0, -1.2]
            # float_rhs: [1.5, 2.0, -3.2]
            # Output tensor: [2.0, -2.0, 2.0]
            result = subtract(float_lhs, float_rhs)
        """
        return self.eltwise_proxy(
            torch.subtract, ttir.SubtractOp, [in0, in1], unit_attrs=unit_attrs
        )

    def remainder(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Elementwise remainder operation.

        The `remainder` operation performs an elementwise remainder (modulo) operation
        between two tensors. For each pair of corresponding elements, it computes the
        remainder when dividing the element in the first tensor (dividend) by the
        element in the second tensor (divisor) and places the result in the output tensor.

        Args:
            in0: First input tensor (dividend)
            in1: Second input tensor (divisor)
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the elementwise remainder of the inputs

        Example:
            # Remainder operation
            # Input tensors:
            # lhs: [17, -17, 17, -17]  # Dividends
            # rhs: [3, 3, -3, -3]      # Divisors
            # Output tensor: [2, -2, 2, -2]
            result = remainder(lhs, rhs)

            # Example with floating point values
            # Input tensors:
            # float_lhs: [10.5, -10.5, 3.0]
            # float_rhs: [3.0, 3.0, 2.0]
            # Output tensor: [1.5, -1.5, 1.0]
            result = remainder(float_lhs, float_rhs)
        """
        return self.eltwise_proxy(
            torch.remainder, ttir.RemainderOp, [in0, in1], unit_attrs=unit_attrs
        )

    def pow(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Elementwise power operation.

        The `pow` operation performs an elementwise exponentiation between two tensors.
        For each pair of corresponding elements, it raises the element in the first
        tensor (base) to the power of the element in the second tensor (exponent) and
        places the result in the output tensor.

        Args:
            in0: First input tensor (base)
            in1: Second input tensor (exponent)
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the elementwise power of the inputs

        Example:
            # Power operation
            # Input tensors:
            # lhs: [2.0, 3.0, 4.0]  # Bases
            # rhs: [2.0, 2.0, 0.5]  # Exponents
            # Output tensor: [4.0, 9.0, 2.0]
            result = pow(lhs, rhs)

            # Example with integer values
            # Input tensors:
            # int_lhs: [2, 3, 5]
            # int_rhs: [3, 2, 1]
            # Output tensor: [8, 9, 5]
            result = pow(int_lhs, int_rhs)
        """
        return self.eltwise_proxy(
            torch.pow, ttir.PowOp, [in0, in1], unit_attrs=unit_attrs
        )

    # class TTIR_ReductionOp

    def argmax(
        self,
        in0: Operand,
        dim_arg: List[int],
        keep_dim: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Argmax reduction operation.

        Determine the indices of the maximum values along a specified dimension of a
        tensor or over all elements in a tensor. This operation reduces the input tensor
        by finding the index of the maximum value along the dimensions specified in
        dim_arg. If dim_arg is not provided, the argmax is computed over all dimensions,
        resulting in a scalar index. If keep_dim is set to true, the reduced dimensions
        are retained with a size of 1.

        Args:
            in0: Input tensor
            dim_arg: List of dimensions to reduce along
            keep_dim: If True, retain reduced dimensions with size 1
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the indices of maximum values along the specified dimensions

        Example:
            # Argmax along dimension 1
            # Input tensor:
            # [[1.0, 5.0, 3.0],
            #  [2.0, 4.0, 6.0]]
            # dim_arg = [1], keep_dim = False
            # Output tensor: [1, 2]  # Index of maximum value in each row (5.0 in first row, 6.0 in second row)
            result = argmax(input_tensor, dim_arg=[1], keep_dim=False)
        """
        kwargs = {"dim_arg": dim_arg, "keep_dim": keep_dim}
        return self.op_proxy(
            self.argmax_golden_function,
            ttir.ArgMaxOp,
            [in0],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            output_type=IntegerType.get_signless(32, self._ctx),
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def argmax_golden_function(
        self, in0: Operand, dim_arg: List[int], keep_dim: bool = False
    ) -> OpView:
        in1 = torch.argmax(in0, dim=dim_arg[0], keepdim=keep_dim)
        return in1.to(torch.int32)

    def sum(
        self,
        in0: Operand,
        dim_arg: List[int] = [0],
        keep_dim: bool = True,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Sum reduction operation.

        The sum operation computes the sum of elements along specified dimensions of
        the input tensor. This operation reduces the input tensor by computing the
        sum of all elements along the dimensions specified in dim_arg. If dim_arg
        is not provided, the sum is computed over all dimensions, resulting in a
        scalar value. If keep_dim is set to true, the reduced dimensions are retained
        with a size of 1.

        Mathematical definition: sum(x, dim) = ∑ x[i] for all i in dimension dim

        Args:
            in0: Input tensor
            dim_arg: List of dimensions to reduce along. Default is [0]
            keep_dim: If True, retain reduced dimensions with size 1. Default is True
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the sum of elements along the specified dimensions

        Example:
            # Sum along dimension 1
            # Input tensor:
            # [[1.0, 2.0, 3.0],
            #  [4.0, 5.0, 6.0]]
            # dim_arg = [1], keep_dim = False
            # Output tensor: [6.0, 15.0]  # Sum of each row
            result = sum(input_tensor, dim_arg=[1], keep_dim=False)

            # Sum along dimension 0
            # Input tensor:
            # [[1.0, 2.0, 3.0],
            #  [4.0, 5.0, 6.0]]
            # dim_arg = [0], keep_dim = False
            # Output tensor: [5.0, 7.0, 9.0]  # Sum of each column
            result = sum(input_tensor, dim_arg=[0], keep_dim=False)

            # Sum over all dimensions
            # Input tensor:
            # [[1.0, 2.0, 3.0],
            #  [4.0, 5.0, 6.0]]
            # keep_dim = False
            # Output tensor: 21.0  # Sum of all elements
            result = sum(input_tensor, keep_dim=False)
        """
        return self.op_proxy(
            torch.sum,
            ttir.SumOp,
            [in0],
            golden_kwargs={"dim": dim_arg, "keepdim": keep_dim},
            ttir_kwargs={"dim_arg": dim_arg, "keep_dim": keep_dim},
            unit_attrs=unit_attrs,
        )

    def mean(
        self,
        in0: Operand,
        dim_arg: List[int] = [0],
        keep_dim: bool = True,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Mean reduction operation.

        The `mean` operation computes the arithmetic mean of elements along specified
        dimensions of the input tensor. This operation reduces the input tensor by
        computing the average of all elements along the dimensions specified in
        dim_arg. If dim_arg is not provided, the mean is computed over all dimensions,
        resulting in a scalar value. If keep_dim is set to true, the reduced dimensions
        are retained with a size of 1.

        Note: For integer input tensors, the result is typically rounded to the nearest
        integer according to the rounding mode.

        Mathematical definition: mean(x, dim) = (∑ x[i]) / n for all i in dimension dim,
        where n is the number of elements in dimension dim

        Args:
            in0: Input tensor
            dim_arg: List of dimensions to reduce along. Default is [0]
            keep_dim: If True, retain reduced dimensions with size 1. Default is True
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the mean of elements along the specified dimensions

        Example:
            # Mean along dimension 1
            # Input tensor:
            # [[1.0, 2.0, 3.0],
            #  [4.0, 5.0, 6.0]]
            # dim_arg = [1], keep_dim = False
            # Output tensor: [2.0, 5.0]  # Mean of each row
            result = mean(input_tensor, dim_arg=[1], keep_dim=False)

            # Mean along dimension 0
            # Input tensor:
            # [[1.0, 2.0, 3.0],
            #  [4.0, 5.0, 6.0]]
            # dim_arg = [0], keep_dim = False
            # Output tensor: [2.5, 3.5, 4.5]  # Mean of each column
            result = mean(input_tensor, dim_arg=[0], keep_dim=False)

            # Mean over all dimensions
            # Input tensor:
            # [[1.0, 2.0, 3.0],
            #  [4.0, 5.0, 6.0]]
            # keep_dim = False
            # Output tensor: 3.5  # Mean of all elements
            result = mean(input_tensor, keep_dim=False)
        """
        return self.op_proxy(
            torch.mean,
            ttir.MeanOp,
            [in0],
            golden_kwargs={"dim": dim_arg, "keepdim": keep_dim},
            ttir_kwargs={"dim_arg": dim_arg, "keep_dim": keep_dim},
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def max(
        self,
        in0: Operand,
        dim_arg: int = None,
        keep_dim: bool = True,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Maximum reduction operation.

        The `max` operation computes the maximum value of elements along specified
        dimensions of the input tensor. This operation reduces the input tensor by
        finding the maximum value of all elements along the dimensions specified in
        dim_arg. If dim_arg is not provided, the maximum is computed over all dimensions,
        resulting in a scalar value. If keep_dim is set to true, the reduced dimensions
        are retained with a size of 1.

        Note: When comparing with NaN values, NaN is typically not selected as the maximum value.

        Mathematical definition: max(x, dim) = max(x[i]) for all i in dimension dim

        Args:
            in0: Input tensor
            dim_arg: Dimension to reduce along. If None, reduce over all dimensions
            keep_dim: If True, retain reduced dimensions with size 1. Default is True
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the maximum values along the specified dimension

        Example:
            # Maximum along dimension 1
            # Input tensor:
            # [[1.0, 5.0, 3.0],
            #  [4.0, 2.0, 6.0]]
            # dim_arg = 1, keep_dim = False
            # Output tensor: [5.0, 6.0]  # Maximum of each row
            result = max(input_tensor, dim_arg=1, keep_dim=False)

            # Maximum along dimension 0
            # Input tensor:
            # [[1.0, 5.0, 3.0],
            #  [4.0, 2.0, 6.0]]
            # dim_arg = 0, keep_dim = False
            # Output tensor: [4.0, 5.0, 6.0]  # Maximum of each column
            result = max(input_tensor, dim_arg=0, keep_dim=False)

            # Maximum over all dimensions
            # Input tensor:
            # [[1.0, 5.0, 3.0],
            #  [4.0, 2.0, 6.0]]
            # dim_arg = None, keep_dim = False
            # Output tensor: 6.0  # Maximum of all elements
            result = max(input_tensor, keep_dim=False)
        """
        # Handle ttir and golden function arguments for edge cases
        golden_kwargs = {}
        ttir_kwargs = {"keep_dim": keep_dim}
        output_shape = [1] * len(self.get_shape(in0))
        if dim_arg:
            golden_kwargs = {"dim": dim_arg, "keepdim": keep_dim}
            ttir_kwargs["dim_arg"] = [dim_arg]
        if not keep_dim:
            output_shape = torch.Size([1])

        return self.op_proxy(
            torch.max,
            ttir.MaxOp,
            [in0],
            golden_kwargs=golden_kwargs,
            ttir_kwargs=ttir_kwargs,
            output_shape=output_shape,
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def min(
        self,
        in0: Operand,
        dim_arg: int = None,
        keep_dim: bool = True,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Minimum reduction operation.

        The `min` operation computes the minimum value of elements along specified
        dimensions of the input tensor. This operation reduces the input tensor by
        finding the minimum value of all elements along the dimensions specified in
        dim_arg. If dim_arg is not provided, the minimum is computed over all dimensions,
        resulting in a scalar value. If keep_dim is set to true, the reduced dimensions
        are retained with a size of 1.

        Note: When comparing with NaN values, NaN is typically not selected as the minimum value.

        Mathematical definition: min(x, dim) = min(x[i]) for all i in dimension dim

        Args:
            in0: Input tensor
            dim_arg: Dimension to reduce along. If None, reduce over all dimensions
            keep_dim: If True, retain reduced dimensions with size 1. Default is True
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the minimum values along the specified dimension

        Example:
            # Minimum along dimension 1
            # Input tensor:
            # [[1.0, 5.0, 3.0],
            #  [4.0, 2.0, 6.0]]
            # dim_arg = 1, keep_dim = False
            # Output tensor: [1.0, 2.0]  # Minimum of each row
            result = min(input_tensor, dim_arg=1, keep_dim=False)

            # Minimum along dimension 0
            # Input tensor:
            # [[1.0, 5.0, 3.0],
            #  [4.0, 2.0, 6.0]]
            # dim_arg = 0, keep_dim = False
            # Output tensor: [1.0, 2.0, 3.0]  # Minimum of each column
            result = min(input_tensor, dim_arg=0, keep_dim=False)

            # Minimum over all dimensions
            # Input tensor:
            # [[1.0, 5.0, 3.0],
            #  [4.0, 2.0, 6.0]]
            # dim_arg = None, keep_dim = False
            # Output tensor: 1.0  # Minimum of all elements
            result = min(input_tensor, keep_dim=False)
        """
        # Handle ttir and golden function arguments for edge cases
        golden_kwargs = {}
        ttir_kwargs = {"keep_dim": keep_dim}
        output_shape = [1] * len(self.get_shape(in0))
        if dim_arg:
            golden_kwargs = {"dim": dim_arg, "keepdim": keep_dim}
            ttir_kwargs["dim_arg"] = [dim_arg]
        if not keep_dim:
            output_shape = torch.Size([1])

        return self.op_proxy(
            torch.min,
            ttir.MinOp,
            [in0],
            golden_kwargs=golden_kwargs,
            ttir_kwargs=ttir_kwargs,
            output_shape=output_shape,
            unit_attrs=unit_attrs,
        )

    # NOTE: Not useable. Boolean tensors are not supported by the runtime. Issue #1775
    def reduce_and(
        self,
        in0: Operand,
        keep_dim: bool = True,
        dim_args: Optional[List] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Logical AND reduction operation.

        The `reduce_and` operation performs a logical AND reduction along specified
        dimensions of the input tensor. This operation reduces the input tensor by
        applying a logical AND operation to all elements along the dimensions
        specified in dim_args. If dim_args is not provided, the reduction is computed
        over all dimensions, resulting in a scalar value. If keep_dim is set to true,
        the reduced dimensions are retained with a size of 1.

        The operation treats non-zero values as True and zero values as False when
        performing the logical AND.

        Mathematical definition: reduce_and(x, dim) = AND(x[i]) for all i in dimension dim

        Args:
            in0: Input tensor
            keep_dim: If True, retain reduced dimensions with size 1. Default is True
            dim_args: List of dimensions to reduce along. Default is None
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the logical AND of elements along the specified dimensions

        Example:
            # Logical AND reduction along dimension 0
            # Input tensor (where 1 represents True and 0 represents False):
            # [[1, 0, 1, 0],
            #  [1, 1, 1, 1],
            #  [0, 0, 1, 1],
            #  [0, 1, 1, 0]]
            # dim_args = [0], keep_dim = False
            # Output tensor: [0, 0, 1, 0]  # Logical AND of each column
            result = reduce_and(input_tensor, dim_args=[0], keep_dim=False)

            # Logical AND reduction along dimension 1
            # Input tensor:
            # [[1, 0, 1, 0],
            #  [1, 1, 1, 1],
            #  [0, 0, 1, 1],
            #  [0, 1, 1, 0]]
            # dim_args = [1], keep_dim = False
            # Output tensor: [0, 1, 0, 0]  # Logical AND of each row
            result = reduce_and(input_tensor, dim_args=[1], keep_dim=False)
        """
        return self.op_proxy(
            torch.all,
            ttir.ReduceAndOp,
            [in0],
            golden_kwargs={"dim": tuple(dim_args), "keepdim": keep_dim},
            ttir_kwargs={"dim_arg": dim_args, "keep_dim": keep_dim},
            unit_attrs=unit_attrs,
        )

    # NOTE: Not useable. Boolean tensors are not supported by the runtime. Issue #1775
    def reduce_or(
        self,
        in0: Operand,
        keep_dim: bool = True,
        dim_args: Optional[List] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Logical OR reduction operation.

        The `reduce_or` operation performs a logical OR reduction along specified
        dimensions of the input tensor. This operation reduces the input tensor by
        applying a logical OR operation to all elements along the dimensions
        specified in dim_args. If dim_args is not provided, the reduction is computed
        over all dimensions, resulting in a scalar value. If keep_dim is set to true,
        the reduced dimensions are retained with a size of 1.

        The operation treats non-zero values as True and zero values as False when
        performing the logical OR.

        Mathematical definition: reduce_or(x, dim) = OR(x[i]) for all i in dimension dim

        Args:
            in0: Input tensor
            keep_dim: If True, retain reduced dimensions with size 1. Default is True
            dim_args: List of dimensions to reduce along. Default is None
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the logical OR of elements along the specified dimensions

        Example:
            # Logical OR reduction along dimension 0
            # Input tensor (where 1 represents True and 0 represents False):
            # [[1, 0, 0, 0],
            #  [1, 1, 0, 1],
            #  [0, 0, 0, 1],
            #  [0, 0, 0, 0]]
            # dim_args = [0], keep_dim = False
            # Output tensor: [1, 1, 0, 1]  # Logical OR of each column
            result = reduce_or(input_tensor, dim_args=[0], keep_dim=False)

            # Logical OR reduction along dimension 1
            # Input tensor:
            # [[1, 0, 0, 0],
            #  [1, 1, 0, 1],
            #  [0, 0, 0, 1],
            #  [0, 0, 0, 0]]
            # dim_args = [1], keep_dim = False
            # Output tensor: [1, 1, 1, 0]  # Logical OR of each row
            result = reduce_or(input_tensor, dim_args=[1], keep_dim=False)
        """
        return self.op_proxy(
            torch.any,
            ttir.ReduceOrOp,
            [in0],
            golden_kwargs={"dim": tuple(dim_args)},
            ttir_kwargs={"dim_arg": dim_args, "keep_dim": keep_dim},
            unit_attrs=unit_attrs,
        )

    def prod(
        self,
        in0: Operand,
        dim_arg: List[int],
        keep_dim: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Product reduction operation.

        The `prod` operation computes the product of elements along specified dimensions
        of the input tensor. This operation reduces the input tensor by multiplying
        all elements along the dimensions specified in dim_arg. If dim_arg is not
        provided, the product is computed over all dimensions, resulting in a scalar
        value. If keep_dim is set to true, the reduced dimensions are retained with
        a size of 1.

        Note: For floating-point inputs, the order of multiplication may affect the
        result due to floating-point precision issues.

        Mathematical definition: prod(x, dim) = ∏ x[i] for all i in dimension dim

        Args:
            in0: Input tensor
            dim_arg: List of dimensions to reduce along
            keep_dim: If True, retain reduced dimensions with size 1. Default is False
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the product of elements along the specified dimensions

        Example:
            # Product along dimension 0
            # Input tensor:
            # [[1, 2, 3],
            #  [4, 5, 6]]
            # dim_arg = [0], keep_dim = False
            # Output tensor: [4, 10, 18]  # Product of each column
            result = prod(input_tensor, dim_arg=[0], keep_dim=False)

            # Product along dimension 1
            # Input tensor:
            # [[1, 2, 3],
            #  [4, 5, 6]]
            # dim_arg = [1], keep_dim = False
            # Output tensor: [6, 120]  # Product of each row
            result = prod(input_tensor, dim_arg=[1], keep_dim=False)
        """
        golden_kwargs = {}
        if len(dim_arg) == 1:
            golden_kwargs["dim"] = dim_arg[0]
            golden_kwargs["keepdim"] = keep_dim
            golden_function = torch.prod
        else:
            golden_function = lambda i: torch.tensor([torch.prod(i[0]).item()])
        return self.op_proxy(
            golden_function,
            ttir.ProdOp,
            [in0],
            golden_kwargs=golden_kwargs,
            ttir_kwargs={"keep_dim": keep_dim, "dim_arg": dim_arg},
            unit_attrs=unit_attrs,
        )

    def embedding(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Embedding lookup operation.

        The `embedding` operation performs a lookup in an embedding table (weight matrix)
        using integer indices. This operation takes an input tensor of indices and a
        weight tensor representing the embedding table. For each index in the input
        tensor, it retrieves the corresponding row from the weight tensor. The result
        is a tensor where each input index is replaced by its corresponding embedding
        vector.

        Note: The indices in the input tensor must be valid indices into the first
        dimension of the weight tensor.

        Args:
            in0: Input tensor containing indices
            in1: Weight tensor (embedding table)
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the embeddings for each input index

        Example:
            # Input tensor (indices):
            # [[0, 2, 5],
            #  [7, 1, 9]]
            #
            # Weight tensor (embedding table):
            # [[0.1, 0.2, 0.3, 0.4],  # embedding vector for index 0
            #  [0.5, 0.6, 0.7, 0.8],  # embedding vector for index 1
            #  [0.9, 1.0, 1.1, 1.2],  # embedding vector for index 2
            #  ...]
            #
            # Output tensor:
            # [[[0.1, 0.2, 0.3, 0.4],  # embedding for index 0
            #   [0.9, 1.0, 1.1, 1.2],  # embedding for index 2
            #   [...]], # embedding for index 5
            #  [[...], # embedding for index 7
            #   [0.5, 0.6, 0.7, 0.8],  # embedding for index 1
            #   [...]]] # embedding for index 9
            result = embedding(indices, weight_matrix)
        """
        embedding = torch.nn.Embedding.from_pretrained(self._get_golden_tensor(in1))
        golden_typecast = self._get_golden_tensor(in0).to(torch.int32)
        golden_input = torch.clamp(
            golden_typecast, 0, (self._get_golden_tensor(in1).size()[0] - 1)
        )
        return self.op_proxy(
            embedding,
            ttir.EmbeddingOp,
            [in0, in1],
            organize_golden_args=lambda i: (golden_input,),
            unit_attrs=unit_attrs,
        )

    def cumsum(
        self,
        in0: Operand,
        in1: Operand,
        dim: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Cumulative sum operation.

        The cumsum operation computes the cumulative sum of elements along a specified
        dimension of the input tensor. For each position in the output tensor, this
        operation computes the sum of all elements in the input tensor along the
        specified dimension up to and including that position. The shape of the output
        tensor matches the shape of the input tensor.

        Args:
            in0: Input tensor
            in1: Output tensor
            dim: The dimension along which to compute the cumulative sum
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the cumulative sums

        Example:
            # Cumulative sum along dimension 0
            # Input tensor:
            # [[1, 2, 3],
            #  [4, 5, 6]]
            # dim = 0
            # Output tensor:
            # [[1, 2, 3],   # first row remains the same
            #  [5, 7, 9]]   # each element is the sum of the corresponding column up to this point
            result = cumsum(input_tensor, output_tensor, dim=0)

            # Cumulative sum along dimension 1
            # Input tensor:
            # [[1, 2, 3],
            #  [4, 5, 6]]
            # dim = 1
            # Output tensor:
            # [[1, 3, 6],   # each element is the sum of the corresponding row up to this point
            #  [4, 9, 15]]
            result = cumsum(input_tensor, output_tensor, dim=1)
        """
        return self.op_proxy(
            torch.cumsum,
            ttir.CumSumOp,
            [in0, in1],
            golden_kwargs={"dim": dim},
            ttir_kwargs={"dim": dim, "output": in1},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0]),
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            unit_attrs=unit_attrs,
        )

    def softmax(
        self, in0: Operand, dimension: int = 1, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Softmax normalization operation.

        The softmax operation applies the softmax function along a specified dimension
        of the input tensor. The softmax function transforms each element of the input
        tensor to a value between 0 and 1, such that the sum of all elements along the
        specified dimension equals 1. This is commonly used to convert a vector of real
        numbers into a probability distribution.

        The softmax function is defined as:
        softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j in the specified dimension

        Note: For numerical stability, the implementation typically subtracts the maximum
        value in each slice before applying the exponential function.

        Args:
            in0: Input tensor
            dimension: The dimension along which to apply the softmax function. Default is 1
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The tensor after applying the softmax function

        Example:
            # Softmax along dimension 1
            # Input tensor:
            # [[1.0, 2.0, 3.0],
            #  [4.0, 1.0, 2.0]]
            # Output tensor (approximate values):
            # [[0.09, 0.24, 0.67],  # sum = 1.0
            #  [0.71, 0.09, 0.20]]  # sum = 1.0
            result = softmax(input_tensor, dimension=1)
        """
        return self.op_proxy(
            # torch.softmax,
            torch.nn.functional.softmax,
            ttir.SoftmaxOp,
            [in0],
            golden_kwargs={"dim": dimension},
            organize_ttir_args=lambda i, o, _: (
                self._get_type(o),
                i[0],
                o,
                dimension,
            ),
            unit_attrs=unit_attrs,
        )

    def transpose(
        self,
        in0: Operand,
        dim0: int = 0,
        dim1: int = 1,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Tensor transpose operation.

        The transpose operation swaps two dimensions of a tensor. This operation
        exchanges the positions of two specified dimensions in the input tensor,
        effectively transposing those dimensions. The shape of the output tensor is
        the same as the input tensor, except that the dimensions specified by dim0
        and dim1 are swapped.

        Args:
            in0: Input tensor
            dim0: First dimension to swap. Default is 0
            dim1: Second dimension to swap. Default is 1
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The transposed tensor

        Example:
            # Transpose dimensions 0 and 1
            # Input tensor:
            # [[[1, 2],
            #   [3, 4],
            #   [5, 6]],
            #  [[7, 8],
            #   [9, 10],
            #   [11, 12]]]
            # Shape: (2, 3, 2)
            #
            # Output tensor:
            # [[[1, 2],
            #   [7, 8]],
            #  [[3, 4],
            #   [9, 10]],
            #  [[5, 6],
            #   [11, 12]]]
            # Shape: (3, 2, 2)
            result = transpose(input_tensor, dim0=0, dim1=1)
        """
        kwargs = {"dim0": dim0, "dim1": dim1}
        return self.op_proxy(
            torch.transpose,
            ttir.TransposeOp,
            [in0],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def concat(
        self, ins: List[Operand], dim: int = 0, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Tensor concatenation operation.

        The `concat` operation joins multiple tensors along a specified dimension. This
        operation concatenates a list of tensors along the dimension specified by dim.
        All input tensors must have the same shape except for the dimension being
        concatenated, and the output tensor's shape will match the input tensors
        except for the concatenated dimension, which will be the sum of the input
        dimensions.

        Args:
            ins: List of input tensors to concatenate
            dim: The dimension along which to concatenate the tensors. Default is 0
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The concatenated tensor

        Example:
            # Concatenate along dimension 0
            # Input1 shape: [2, 3]
            # Input2 shape: [3, 3]
            # Output shape: [5, 3]
            result = concat([input1, input2], dim=0)

            # Concatenate along dimension 1
            # Input1 shape: [2, 3]
            # Input2 shape: [2, 2]
            # Output shape: [2, 5]
            result = concat([input1, input2], dim=1)
        """
        kwargs = {"dim": dim}
        return self.op_proxy(
            torch.concat,
            ttir.ConcatOp,
            ins,
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            # special handling is needed here to get around arg expansion; `torch.concat` takes a tuple of tensors on input
            organize_golden_args=lambda i: (
                tuple([self._get_golden_tensor(i_i) for i_i in i]),
            ),
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i, o),
            unit_attrs=unit_attrs,
        )

    def repeat(
        self, in0: Operand, dims: List[int], unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Repeat operation.

        The `repeat` operation creates a new tensor by replicating the input tensor's
        elements along specified dimensions. This operation repeats the entire input
        tensor along each dimension according to the values specified in dims. The
        resulting tensor's shape is the product of the input tensor's shape and the
        corresponding repeat values.

        Args:
            in0: Input tensor
            dims: List of repeat values for each dimension
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The repeated tensor

        Example:
            # Input tensor:
            # [[1, 2],
            #  [3, 4]]
            # Shape: (2, 2)
            #
            # dims = [2, 3]
            # Output tensor:
            # [[1, 2, 1, 2, 1, 2],
            #  [3, 4, 3, 4, 3, 4],
            #  [1, 2, 1, 2, 1, 2],
            #  [3, 4, 3, 4, 3, 4]]
            # Shape: (4, 6) = (2*2, 2*3)
            result = repeat(input_tensor, dims=[2, 3])
        """
        return self.op_proxy(
            torch.Tensor.repeat,
            ttir.RepeatOp,
            [in0],
            golden_kwargs={"repeats": dims},
            ttir_kwargs={"repeat_dimensions": dims},
            unit_attrs=unit_attrs,
        )

    def repeat_interleave(
        self,
        in0: Operand,
        in1: Operand,
        repeats: int,
        dim: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Tensor repeat interleave operation.

        The `repeat_interleave` operation repeats elements of a tensor along a specified
        dimension. Unlike the repeat operation which repeats the entire tensor, this
        operation repeats each individual element of the input tensor the specified
        number of times along the given dimension. This creates an interleaved
        pattern of repeated values.

        Args:
            in0: Input tensor
            in1: Output tensor
            repeats: The number of times to repeat each element
            dim: The dimension along which to repeat elements
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The tensor with repeated elements

        Example:
            # Repeat interleave along dimension 0 with repeats=2
            # Input tensor:
            # [[1.0, 2.0, 3.0],
            #  [4.0, 5.0, 6.0]]
            # Output tensor:
            # [[1.0, 2.0, 3.0],  # First row repeated
            #  [1.0, 2.0, 3.0],
            #  [4.0, 5.0, 6.0],  # Second row repeated
            #  [4.0, 5.0, 6.0]]
            result = repeat_interleave(input_tensor, output_tensor, repeats=2, dim=0)

            # Repeat interleave along dimension 1 with repeats=3
            # Input tensor:
            # [[1.0, 2.0],
            #  [3.0, 4.0]]
            # Output tensor:
            # [[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],  # Each element repeated 3 times
            #  [3.0, 3.0, 3.0, 4.0, 4.0, 4.0]]
            result = repeat_interleave(input_tensor, output_tensor, repeats=3, dim=1)
        """
        return self.op_proxy(
            torch.repeat_interleave,
            ttir.RepeatInterleaveOp,
            [in0, in1],
            golden_kwargs={"repeats": repeats, "dim": dim},
            ttir_kwargs={"repeats": repeats, "dim": dim},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1]),
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            output_type=self.get_type_from_torch_dtype(
                self._get_golden_tensor(in1).dtype
            ),
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def fill_cache(
        self,
        in0: Operand,
        in1: Operand,
        batch_offset: int = 0,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Cache filling operation.

        The `fill_cache` operation fills a cache tensor with values from an input tensor.
        Unlike update_cache which updates specific positions, this operation fills the
        entire cache or a contiguous section of it with values from the input tensor.
        This is commonly used to initialize a cache in sequence models.

        Args:
            in0: Cache tensor to be filled
            in1: Input tensor containing the values to fill the cache with
            batch_offset: Offset in the batch dimension. Default is 0
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The filled cache tensor

        Example:
            # Fill cache with input values
            # Cache tensor shape: (2, 16, 64)  # Batch size 2, sequence length 16, hidden dim 64
            # Input tensor shape: (2, 16, 64)  # Initial values for the entire cache
            result = fill_cache(cache_tensor, input_tensor, batch_offset=0)
            # The entire cache tensor is filled with values from input

            # Fill a portion of the cache
            # Cache tensor shape: (2, 16, 64)  # Batch size 2, sequence length 16, hidden dim 64
            # Input tensor shape: (2, 8, 64)   # Values for half of the cache
            result = fill_cache(cache_tensor, input_tensor, batch_offset=0)
            # The first 8 positions of the cache are filled with values from input
        """
        cache_tensor = self._get_golden_tensor(in0)
        input_tensor = self._get_golden_tensor(in1)
        cache_tensor[:, :, : input_tensor.shape[2], :] = input_tensor
        return self.op_proxy(
            torch.clone,
            ttir.FillCacheOp,
            [in0, in1],
            golden_kwargs={"input": cache_tensor},
            ttir_kwargs={"batch_offset": batch_offset},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1]),
            organize_golden_args=lambda i: 0,
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def update_cache(
        self,
        in0: Operand,
        in1: Operand,
        in2: Operand,
        batch_offset: int = 0,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Cache update operation.

        The `update_cache` operation updates a cache tensor with values from an input
        tensor at specific indices. This operation is commonly used in sequence models
        like transformers to update a key-value cache with new token information. It
        takes a cache tensor, an input tensor, and update indices, and updates the
        cache at the specified positions.

        Args:
            in0: Cache tensor to be updated
            in1: Input tensor containing new values
            in2: Indices tensor specifying where to update the cache
            batch_offset: Offset in the batch dimension. Default is 0
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The updated cache tensor

        Example:
            # Update cache at specific indices
            # Cache tensor shape: (2, 16, 64)  # Batch size 2, sequence length 16, hidden dim 64
            # Input tensor shape: (2, 1, 64)   # New token embeddings
            # Update index: [15]               # Update at position 15
            result = update_cache(cache_tensor, input_tensor, update_index, batch_offset=0)
            # The cache tensor is updated at position 15 for both batches with the values from input
        """
        cache = self._get_golden_tensor(in0)
        input_tensor = self._get_golden_tensor(in1)
        index = torch.clamp(self._get_golden_tensor(in2), 0, cache.size()[2])
        a = cache[:, :, : index[0], :]
        b = cache[:, :, : (cache.size()[2] - index[0] - 1), :]

        return self.op_proxy(
            torch.cat,
            ttir.UpdateCacheOp,
            [in0, in1, in2],
            golden_kwargs={"tensors": (a, input_tensor, b), "dim": 2},
            ttir_kwargs={"batch_offset": batch_offset},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1], i[2]),
            organize_golden_args=lambda i: 0,
            unit_attrs=unit_attrs,
        )

    def broadcast(
        self,
        in0: Operand,
        in1: Operand,
        broadcast_dimensions: List[int],
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Broadcast operation.

        The broadcast operation expands the dimensions of an input tensor according to
        specified broadcast dimensions. This operation takes an input tensor and
        broadcasts it to a larger shape by repeating elements along dimensions where
        the input has size 1 and the output has a larger size. This is commonly used
        to make tensors compatible for elementwise operations.

        Note: Currently, when generating a TTNN executable, the broadcast and repeat
        operations share the same semantics due to the lack of tensor view support
        in TTNN. As a result, the broadcast operation is lowered to a repeat
        operation in the TTNN compilation pipeline.

        Args:
            in0: Input tensor to broadcast
            in1: Output tensor
            broadcast_dimensions: The number of times to broadcast the tensor along each dimension
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The broadcasted tensor

        Example:
            # Broadcast a tensor from shape [1, 1, 32] to [1, 16, 32]
            # Input tensor shape: (1, 1, 32)
            # Output tensor shape: (1, 16, 32)
            result = broadcast(input_tensor, output_tensor, broadcast_dimensions=[1, 16, 1])
            # The input tensor is repeated 16 times along the second dimension

            # Broadcast a tensor from shape [1, 3] to [2, 3]
            # Input tensor shape: (1, 3)
            # Output tensor shape: (2, 3)
            result = broadcast(input_tensor, output_tensor, broadcast_dimensions=[2, 1])
            # The input tensor is repeated 2 times along the first dimension
        """
        return self.op_proxy(
            torch.broadcast_to,
            ttir.BroadcastOp,
            [in0],
            golden_kwargs={"size": self.get_shape(in1)},
            ttir_kwargs={"broadcast_dimensions": broadcast_dimensions},
            unit_attrs=unit_attrs,
        )

    def conv2d(
        self,
        in0: Operand,
        weight: Operand,
        bias: Optional[Operand],
        in1: Operand,
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """2D convolution operation.

        Applies a 2D convolution over an input image composed of several input planes.
        This operation performs a 2D convolution on the input tensor using the provided
        weight tensor and optional bias. It supports configurable stride, padding,
        dilation, and grouping parameters to control the convolution behavior.

        Args:
            in0: Input tensor in format (N, H_in, W_in, C) where:
                - N is the batch size
                - H_in is the height of the input planes
                - W_in is the width of the input planes
                - C is the number of channels
            weight: Weight tensor in format (O, C/G, K_H, K_W) where:
                - C is the number of input channels
                - O is the number of output channels
                - G is the number of groups
                - K_H is the height of the kernel
                - K_W is the width of the kernel
            bias: Optional bias tensor in format (1, 1, 1, O)
            in1: Output tensor in format (N, H_out, W_out, O) where:
                - H_out = (H_in + pT + pB - dH * (K_H - 1) - 1) / sH + 1
                - W_out = (W_in + pL + pR - dW * (K_W - 1) - 1) / sW + 1
            stride: Stride for height and width dimensions. Can be:
                - int: Same stride for height and width (sH = sW = value)
                - List[int]: [sH, sW] where sH is stride for height and sW for width
            padding: Padding configuration. Can be:
                - int: Same padding for all sides (pT = pL = pB = pR = value)
                - List[int] (length 2): [pH, pW] where pH is padding for height
                  (top/bottom) and pW for width (left/right)
                - List[int] (length 4): [pT, pL, pB, pR] for top, left, bottom,
                  and right padding respectively
            dilation: Spacing between kernel elements. Can be:
                - int: Same dilation for height and width (dH = dW = value)
                - List[int]: [dH, dW] where dH is dilation for height and dW for width
            groups: Number of blocked connections from input channels to output channels.
                   Input and output channels must both be divisible by groups
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The output tensor after 2D convolution

        Example:
            # Basic 2D convolution
            # Input: (1, 28, 28, 3)    # Batch size 1, 28x28 image, 3 channels
            # Weight: (16, 3, 3, 3)    # 16 output channels, 3 input channels, 3x3 kernel
            # Bias: (1, 1, 1, 16)      # Bias for 16 output channels
            # Output: (1, 26, 26, 16)  # Output shape with no padding
            result = conv2d(input_tensor, weight_tensor, bias_tensor, output_tensor,
                          stride=1, padding=0, dilation=1, groups=1)

            # Convolution with stride 2 and padding
            # Input: (1, 28, 28, 3)    # Batch size 1, 28x28 image, 3 channels
            # Weight: (16, 3, 3, 3)    # 16 output channels, 3 input channels, 3x3 kernel
            # Bias: (1, 1, 1, 16)      # Bias for 16 output channels
            # Output: (1, 14, 14, 16)  # Output shape with stride 2
            result = conv2d(input_tensor, weight_tensor, bias_tensor, output_tensor,
                          stride=[2, 2], padding=[1, 1, 1, 1], dilation=[1, 1], groups=1)
        """
        if not bias:
            bias = None
        return self.op_proxy(
            self.conv2d_golden_function,
            ttir.Conv2dOp,
            [in0, weight, bias],
            golden_kwargs={
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
            },
            ttir_kwargs={
                "stride": (
                    IntegerAttr.get(IntegerType.get_signed(32), stride)
                    if isinstance(stride, int)
                    else DenseI32ArrayAttr.get(stride)
                ),
                "padding": (
                    IntegerAttr.get(IntegerType.get_signed(32), padding)
                    if isinstance(padding, int)
                    else DenseI32ArrayAttr.get(padding)
                ),
                "dilation": (
                    IntegerAttr.get(IntegerType.get_signed(32), dilation)
                    if isinstance(dilation, int)
                    else DenseI32ArrayAttr.get(dilation)
                ),
                "groups": groups,
            },
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1], o),
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def conv2d_golden_function(
        self,
        input_tensor: Operand,
        weight: Operand,
        bias: Optional[Operand],
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int,
    ) -> Operand:
        # Reorganize ttir_kwargs into golden_kwargs
        stride = list(stride) if not isinstance(stride, int) else int(stride)
        padding = list(padding) if not isinstance(padding, int) else int(padding)
        dilation = list(dilation) if not isinstance(dilation, int) else int(dilation)

        # ttir can handle a broadcastable bias in the shape [1, 1, 1, C_out], but PyTorch requires the bias is rank 1: [C_out]
        bias = bias.squeeze()  # Removes all dims of size 1

        # Reorganize input and output tensors, golden and ttir functions have different expected tensor shapes
        input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)
        result = torch.nn.functional.conv2d(
            input_tensor,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        result = result.transpose(-3, -2).transpose(-2, -1)
        return result

    def conv_transpose2d(
        self,
        in0: Operand,
        weight: Operand,
        bias: Optional[Operand],
        in1: Operand,
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        output_padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """2D transposed convolution operation.

        Applies a 2D transposed convolution operator over an input image composed of
        several input planes. This operation performs the gradient of a 2D convolution
        with respect to the input, which is useful for tasks like upsampling feature
        maps in neural networks. It supports configurable stride, padding, dilation,
        output padding, and grouping parameters.

        Args:
            in0: Input tensor in format (N, H_in, W_in, C) where:
                - N is the batch size
                - H_in is the height of the input planes
                - W_in is the width of the input planes
                - C is the number of channels
            weight: Weight tensor in format (C, O/G, K_H, K_W) where:
                - C is the number of input channels
                - O is the number of output channels
                - G is the number of groups
                - K_H is the height of the kernel
                - K_W is the width of the kernel
            bias: Optional bias tensor in format (1, 1, 1, O)
            in1: Output tensor in format (N, H_out, W_out, O) where:
                - H_out = (H_in - 1) * stride[0] - (padding_top + padding_bottom)
                  + dilation[0] * (K_H - 1) + output_padding[0] + 1
                - W_out = (W_in - 1) * stride[1] - (padding_left + padding_right)
                  + dilation[1] * (K_W - 1) + output_padding[1] + 1
            stride: Controls the stride for the cross-correlation. Can be:
                - int: Same stride for height and width
                - List[int]: [sH, sW] where sH is stride for height and sW for width
            padding: Controls the implicit zero padding on both sides. Can be:
                - int: Same padding for all sides
                - List[int] (length 2): [pH, pW] for height and width padding
                - List[int] (length 4): [pT, pL, pB, pR] for top, left, bottom,
                  and right padding
            output_padding: Controls additional size added to output shape. Can be:
                - int: Same output padding for height and width
                - List[int]: [opH, opW] for height and width output padding
            dilation: Controls spacing between kernel points. Can be:
                - int: Same dilation for height and width
                - List[int]: [dH, dW] for height and width dilation
            groups: Controls connections between inputs and outputs. Must be
                   divisible by input and output channels
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The output tensor after 2D transposed convolution

        Example:
            # Basic 2D transposed convolution
            # Input: (1, 14, 14, 16)   # Batch size 1, 14x14 feature map, 16 channels
            # Weight: (16, 8, 3, 3)    # 16 input channels, 8 output channels, 3x3 kernel
            # Bias: (1, 1, 1, 8)       # Bias for 8 output channels
            # Output: (1, 28, 28, 8)   # Output shape with stride 2
            result = conv_transpose2d(input_tensor, weight_tensor, bias_tensor,
                                    output_tensor, stride=[2, 2], padding=[0, 0, 0, 0],
                                    output_padding=[0, 0], dilation=[1, 1], groups=1)

            # Transposed convolution with padding and output padding
            # Input: (1, 14, 14, 16)   # Batch size 1, 14x14 feature map, 16 channels
            # Weight: (16, 8, 4, 4)    # 16 input channels, 8 output channels, 4x4 kernel
            # Bias: (1, 1, 1, 8)       # Bias for 8 output channels
            # Output: (1, 29, 29, 8)   # Output shape with output padding
            result = conv_transpose2d(input_tensor, weight_tensor, bias_tensor,
                                    output_tensor, stride=[2, 2], padding=[1, 1, 1, 1],
                                    output_padding=[1, 1], dilation=[1, 1], groups=1)
        """
        if not bias:
            bias = None
        return self.op_proxy(
            self.conv_transpose2d_golden_function,
            ttir.ConvTranspose2dOp,
            [in0, weight],
            golden_kwargs={
                "stride": stride,
                "padding": padding,
                "output_padding": output_padding,
                "dilation": dilation,
                "groups": groups,
            },
            ttir_kwargs={
                "stride": (
                    IntegerAttr.get(IntegerType.get_signless(32), stride)
                    if isinstance(stride, int)
                    else DenseI32ArrayAttr.get(stride)
                ),
                "padding": (
                    IntegerAttr.get(IntegerType.get_signless(32), padding)
                    if isinstance(padding, int)
                    else DenseI32ArrayAttr.get(padding)
                ),
                "output_padding": (
                    IntegerAttr.get(IntegerType.get_signless(32), output_padding)
                    if isinstance(output_padding, int)
                    else DenseI32ArrayAttr.get(output_padding)
                ),
                "dilation": (
                    IntegerAttr.get(IntegerType.get_signless(32), dilation)
                    if isinstance(dilation, int)
                    else DenseI32ArrayAttr.get(dilation)
                ),
                "groups": (
                    IntegerAttr.get(IntegerType.get_signless(32), groups)
                    if isinstance(groups, int)
                    else DenseI32ArrayAttr.get(groups)
                ),
                "bias": bias,
            },
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def conv_transpose2d_golden_function(
        self,
        input_tensor: Operand,
        weight: Operand,
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        output_padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int,
    ) -> Operand:
        # Reorganize ttir_kwargs into golden_kwargs
        stride = list(stride) if not isinstance(stride, int) else int(stride)
        padding = list(padding) if not isinstance(padding, int) else int(padding)
        output_padding = (
            list(output_padding)
            if not isinstance(output_padding, int)
            else int(output_padding)
        )
        dilation = list(dilation) if not isinstance(dilation, int) else int(dilation)
        golden_bias = torch.rand((weight.size()[0]), dtype=input_tensor.dtype)

        # Reorganize input and output tensors, golden and ttir functions have different expected tensor shapes
        input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)
        result = torch.nn.functional.conv_transpose2d(
            input_tensor,
            weight,
            bias=golden_bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )
        result = result.transpose(-3, -2).transpose(-2, -1)
        return result

    def max_pool2d(
        self,
        in0: Operand,
        in1: Operand,
        kernel_height: int,
        kernel_width: int,
        stride_height: int,
        stride_width: int,
        dilation_height: int,
        dilation_width: int,
        ceil_mode: bool,
        padding_left: int,
        padding_right: int,
        padding_top: int,
        padding_bottom: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """2D maximum pooling operation.

        The `max_pool2d` operation applies a 2D maximum pooling over an input tensor
        composed of several input planes. This operation performs downsampling by
        dividing the input into local regions and computing the maximum value of each
        region. It reduces the spatial dimensions (height and width) of an input
        tensor while preserving the batch and channel dimensions. This is commonly
        used in neural networks to reduce the spatial size of feature maps while
        retaining the most important features.

        Args:
            in0: Input tensor in NHWC format (batch, height, width, channels)
            in1: Output tensor
            kernel_height: Height of the pooling kernel
            kernel_width: Width of the pooling kernel
            stride_height: Stride along the height dimension
            stride_width: Stride along the width dimension
            dilation_height: Dilation factor for height dimension
            dilation_width: Dilation factor for width dimension
            ceil_mode: When true, uses ceil instead of floor for output shape calculation
            padding_left: Padding on the left side
            padding_right: Padding on the right side
            padding_top: Padding on the top side
            padding_bottom: Padding on the bottom side
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: Output tensor after maximum pooling

        Example:
            # Basic 2D max pooling with a 2x2 kernel and stride 1
            # Input tensor shape: (1, 3, 3, 1) with values:
            # [[[1, 2, 3],
            #   [4, 5, 6],
            #   [7, 8, 9]]]
            # Output tensor shape: (1, 2, 2, 1)
            result = max_pool2d(input_tensor, output_tensor,
                              kernel_height=2, kernel_width=2,
                              stride_height=1, stride_width=1,
                              dilation_height=1, dilation_width=1,
                              ceil_mode=False,
                              padding_left=0, padding_right=0,
                              padding_top=0, padding_bottom=0)
            # Result: [[[5, 6],
            #           [8, 9]]]
            # Where: 5 = max(1,2,4,5), 6 = max(2,3,5,6),
            #        8 = max(4,5,7,8), 9 = max(5,6,8,9)
        """
        return self.op_proxy(
            self.max_pool2d_golden_function,
            ttir.MaxPool2dOp,
            [in0],
            golden_kwargs={
                "kernel_size": (kernel_height, kernel_width),
                "stride": (stride_height, stride_width),
                "padding": (padding_top, padding_left),
                "dilation": (dilation_height, dilation_width),
                "ceil_mode": ceil_mode,
            },
            ttir_kwargs={
                "kernel_height": kernel_height,
                "kernel_width": kernel_width,
                "stride_height": stride_height,
                "stride_width": stride_width,
                "dilation_height": dilation_height,
                "dilation_width": dilation_width,
                "ceil_mode": ceil_mode,
                "padding_left": padding_left,
                "padding_right": padding_right,
                "padding_top": padding_top,
                "padding_bottom": padding_bottom,
            },
            unit_attrs=unit_attrs,
        )

    def tilize_golden(self, input: torch.Tensor) -> torch.Tensor:
        """Convert a tensor into a tiled format for efficient computation.

        This is an internal helper function that rearranges a tensor into a tiled
        format by dividing it into tiles of size 32x32, with each tile further
        divided into 16x16 faces. The values are rearranged to optimize memory
        access patterns.

        Args:
            input: Input tensor to be tilized

        Returns:
            torch.Tensor: The tilized tensor with the same shape as input but
                         with values rearranged in a tiled pattern
        """
        shape = input.shape
        TILE_SIZE = 32
        FACE_SIZE = 16
        Y_TILES = shape[0] // TILE_SIZE
        X_TILES = shape[1] // TILE_SIZE
        FACES_PER_TILE = TILE_SIZE // FACE_SIZE

        tilized = torch.zeros((input.numel(),))

        idx = 0
        for tile_y in range(Y_TILES):
            for tile_x in range(X_TILES):
                for face_y in range(FACES_PER_TILE):
                    for face_x in range(FACES_PER_TILE):
                        for datum_y in range(FACE_SIZE):
                            for datum_x in range(FACE_SIZE):
                                tilized[idx] = input[
                                    datum_y + tile_y * TILE_SIZE + face_y * FACE_SIZE,
                                    datum_x + tile_x * TILE_SIZE + face_x * FACE_SIZE,
                                ]
                                idx += 1

        tilized = tilized.reshape(shape)
        return tilized

    def untilize_golden(self, input: torch.Tensor) -> torch.Tensor:
        """Convert a tiled tensor back to its original format.

        This is an internal helper function that reverses the tilize_golden
        operation, taking a tensor in tiled format and rearranging its values
        back to their original positions.

        Args:
            input: Input tensor in tiled format

        Returns:
            torch.Tensor: The untilized tensor with values restored to their
                         original positions
        """
        shape = input.shape
        TILE_SIZE = 32
        FACE_SIZE = 16
        Y_TILES = shape[0] // TILE_SIZE
        X_TILES = shape[1] // TILE_SIZE
        FACES_PER_TILE = TILE_SIZE // FACE_SIZE

        untilized = torch.zeros_like(input)
        flattened = input.flatten()

        idx = 0
        for tile_y in range(Y_TILES):
            for tile_x in range(X_TILES):
                for face_y in range(FACES_PER_TILE):
                    for face_x in range(FACES_PER_TILE):
                        for datum_y in range(FACE_SIZE):
                            for datum_x in range(FACE_SIZE):
                                # Calculate the original position
                                orig_y = (
                                    datum_y + tile_y * TILE_SIZE + face_y * FACE_SIZE
                                )
                                orig_x = (
                                    datum_x + tile_x * TILE_SIZE + face_x * FACE_SIZE
                                )

                                # Place the value from the tilized tensor back to its original position
                                untilized[orig_y, orig_x] = flattened[idx]
                                idx += 1

        return untilized

    @autodoc_skip
    def max_pool2d_golden_function(
        self,
        input_tensor: Operand,
        kernel_size: tuple[int],
        stride: tuple[int],
        padding: tuple[int],
        dilation: tuple[int],
        ceil_mode: bool,
    ):
        # TTIR  max_pool2d is channels last. PyTorch max_pool2d is channels first.
        # We need to transpose the input tensor to channels first before applying max_pool2d,
        # and transpose back to channels last afterward to properly calculate the golden tensor.
        # TTIR  max_pool2d is channels last. PyTorch max_pool2d is channels first.
        # We need to transpose the input tensor to channels first before applying max_pool2d,
        # and transpose back to channels last afterward to properly calculate the golden tensor.
        maxpool_object = torch.nn.MaxPool2d(
            kernel_size, stride, padding, dilation, ceil_mode
        )
        input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)
        result = maxpool_object(input_tensor)
        result = result.transpose(-3, -2).transpose(-2, -1)
        return result

    def reshape(
        self, in0: Operand, shape: Shape, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Tensor reshape operation.

        The reshape operation changes the shape of a tensor without changing the data
        or number of elements. This operation takes an input tensor and reshapes it
        to a new shape specified by the shape attribute. The total number of elements
        in the tensor must remain the same after reshaping. This is commonly used in
        neural networks to change the dimensionality of tensors between layers.

        Args:
            in0: The input tensor to reshape
            shape: The new shape for the tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The reshaped tensor

        Example:
            # Reshape a 2x3 tensor to a 1x6 tensor
            # Input tensor shape: [2, 3]
            # Output tensor shape: [1, 6]
            result = reshape(input_tensor, shape=[1, 6])

            # Reshape a 3D tensor to a 2D tensor
            # Input tensor shape: [2, 3, 4]
            # Output tensor shape: [6, 4]
            result = reshape(input_tensor, shape=[6, 4])

        Note:
            The total number of elements in the input tensor must equal the total
            number of elements in the output tensor. For example, a tensor of shape
            [2,3] (6 elements) can be reshaped to [1,6], [6,1], [2,1,3], etc.,
            but not to [2,4] (8 elements).
        """
        kwargs = {"shape": shape}
        return self.op_proxy(
            torch.reshape,
            ttir.ReshapeOp,
            [in0],
            ttir_kwargs=kwargs,
            golden_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def pad(
        self,
        in0: Operand,
        in1: Operand,
        padding: List[int],
        value: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Tensor padding operation.

        The `pad` operation adds padding to the edges of an input tensor with a
        specified constant value. This operation extends the dimensions of the
        input tensor by adding padding elements with a constant value. The padding
        is specified for each dimension as the number of elements to add at the
        beginning (low) and end (high) of that dimension.

        Args:
            in0: The input tensor to pad
            in1: The output tensor
            padding: The padding values for each dimension, specified as
                    [dim0_low, dim0_high, dim1_low, dim1_high, ...]
            value: The constant value to use for the padding elements
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The padded tensor

        Example:
            # Pad a 2x3 tensor with different padding on each dimension
            # Input tensor with values:
            # [[1, 2, 3],
            #  [4, 5, 6]]
            result = pad(input_tensor, output_tensor,
                        padding=[1, 0, 1, 1],  # [dim0_low, dim0_high, dim1_low, dim1_high]
                        value=0)
            # Result:
            # [[0, 0, 0, 0, 0],
            #  [0, 1, 2, 3, 0],
            #  [0, 4, 5, 6, 0]]

        Note:
            The shape of the output tensor must match the shape of the input tensor
            plus the padding specified in the padding attribute. For example, if the
            input shape is [2,3] and the padding is [1,0,1,1], then the output
            shape must be [3,5].
        """
        # Reformatting padding dimensions for golden tensor:
        golden_padding = []
        for i in range(len(padding) // 2):
            golden_padding.append(padding[-((2 * i) + 2)])
            golden_padding.append(padding[-((2 * i) + 1)])
        return self.op_proxy(
            torch.nn.functional.pad,
            ttir.PadOp,
            [in0, in1],
            golden_kwargs={"pad": golden_padding, "mode": "constant", "value": value},
            ttir_kwargs={"padding": padding, "value": value},
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1]),
            unit_attrs=unit_attrs,
        )

    def select(
        self,
        in0: Operand,
        dim: int = 0,
        begin: int = 0,
        length: int = 2,
        stride: Optional[int] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Tensor selection operation.

        The select operation extracts a sub-tensor (slice) from the input tensor
        along a specified dimension. Unlike the more general slice operation,
        select operates on a single dimension with a specified starting index,
        length, and optional stride. This is useful for extracting specific
        segments of a tensor along a particular axis.

        Args:
            in0: The input tensor to select from
            dim: The dimension along which to select elements
            begin: The starting index for selection
            length: The number of elements to select
            stride: The step size for selection. A value of None means no stride
                   (consecutive elements)
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The selected tensor

        Example:
            # Select elements 2, 3, 4 from a 1D tensor along dimension 0
            # Input tensor: [1, 2, 3, 4, 5, 6]
            result = select(input_tensor, dim=0, begin=2, length=3)
            # Result: [3, 4, 5]

            # Select every other row from a 2D tensor
            # Input tensor:
            # [[1, 2, 3],
            #  [4, 5, 6],
            #  [7, 8, 9],
            #  [10, 11, 12]]
            result = select(input_tensor, dim=0, begin=0, length=2, stride=2)
            # Result:
            # [[1, 2, 3],
            #  [7, 8, 9]]
        """
        end = begin + length - 1
        index = torch.tensor([begin, end])
        # TODO: handle stride. Issue #2488
        if stride:
            pass
        return self.op_proxy(
            torch.index_select,
            ttir.SelectOp,
            [in0],
            golden_kwargs={"dim": dim, "index": index},
            ttir_kwargs={
                "dim": dim,
                "begin": begin,
                "length": length,
                "stride": stride,
            },
            unit_attrs=unit_attrs,
        )

    def index(
        self,
        in0: Operand,
        dim: int,
        begin: int,
        end: int,
        step: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Tensor indexing operation.

        The `index` operation extracts a sub-tensor (slice) from the input tensor
        along a specified dimension. This operation selects elements from the input
        tensor along a single dimension based on the specified begin, end, and step
        indices. It's similar to Python's slicing notation tensor[:, begin:end:step, :]
        where the slicing is applied only to the specified dimension.

        Args:
            in0: The input tensor to index
            dim: The dimension along which to index
            begin: The starting index
            end: The ending index (exclusive)
            step: The step size between indices
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The indexed tensor

        Example:
            # Extract elements with indices 1, 3, 5 from dimension 0 of a 1D tensor
            # Input tensor: [1, 2, 3, 4, 5, 6]
            result = index(input_tensor, dim=0, begin=1, end=6, step=2)
            # Result: [2, 4, 6]

            # Extract columns 0 and 2 from a 2D tensor
            # Input tensor:
            # [[1, 2, 3, 4],
            #  [5, 6, 7, 8],
            #  [9, 10, 11, 12]]
            result = index(input_tensor, dim=1, begin=0, end=3, step=2)
            # Result:
            # [[1, 3],
            #  [5, 7],
            #  [9, 11]]
        """
        import math

        num_indices = math.ceil((end - begin) / step)
        indices = []
        for i in range(num_indices):
            indices.append((begin + i) * step)
        index = torch.tensor(indices)
        return self.op_proxy(
            torch.index_select,
            ttir.IndexOp,
            [in0],
            golden_kwargs={"dim": dim, "index": index},
            ttir_kwargs={"dim": dim, "begin": begin, "end": end, "step": step},
            unit_attrs=unit_attrs,
        )

    def squeeze(
        self,
        in0: Operand,
        dim: Optional[int] = 0,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Tensor dimension squeezing operation.

        The squeeze operation removes a dimension of size 1 from the shape of a
        tensor. This operation is commonly used to eliminate unnecessary singleton
        dimensions from a tensor's shape. It specifies which dimension to remove
        using the dim parameter. The specified dimension must have size 1.

        Args:
            in0: The input tensor to squeeze
            dim: The dimension to squeeze. Must be a dimension with size 1
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The squeezed tensor

        Example:
            # Squeeze dimension 0 from a tensor of shape [1, 3, 4]
            # Input tensor shape: [1, 3, 4]
            result = squeeze(input_tensor, dim=0)
            # Result: tensor with shape [3, 4]

            # Squeeze dimension 1 from a tensor of shape [2, 1, 3]
            # Input tensor shape: [2, 1, 3]
            result = squeeze(input_tensor, dim=1)
            # Result: tensor with shape [2, 3]

        Note:
            The specified dimension must have size 1. The shape of the output
            tensor is the same as the input tensor with the specified dimension
            removed. For example, squeezing dimension 1 of a tensor with shape
            [2, 1, 3] results in a tensor with shape [2, 3].
        """
        kwargs = {"dim": dim}
        return self.op_proxy(
            torch.squeeze,
            ttir.SqueezeOp,
            [in0],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def unsqueeze(
        self,
        in0: Operand,
        dim: Optional[int] = 0,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Tensor dimension insertion operation.

        The unsqueeze operation inserts a dimension of size 1 into the shape of a
        tensor. This operation is the inverse of the squeeze operation and is
        commonly used to add a singleton dimension to a tensor's shape. It
        specifies which position to insert the new dimension using the dim
        parameter.

        Args:
            in0: The input tensor to unsqueeze
            dim: The position to insert the new dimension
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The unsqueezed tensor

        Example:
            # Insert a dimension at position 0 of a tensor with shape [3, 4]
            # Input tensor shape: [3, 4]
            result = unsqueeze(input_tensor, dim=0)
            # Result: tensor with shape [1, 3, 4]

            # Insert a dimension at position 1 of a tensor with shape [2, 3]
            # Input tensor shape: [2, 3]
            result = unsqueeze(input_tensor, dim=1)
            # Result: tensor with shape [2, 1, 3]

        Note:
            The shape of the output tensor is the same as the input tensor with
            a new dimension of size 1 inserted at the specified position. For
            example, unsqueezing at position 1 of a tensor with shape [2, 3]
            results in a tensor with shape [2, 1, 3].
        """
        kwargs = {"dim": dim}
        return self.op_proxy(
            torch.unsqueeze,
            ttir.UnsqueezeOp,
            [in0],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def clamp_scalar(
        self,
        in0: Operand,
        min_arg: Optional[float] = None,
        max_arg: Optional[float] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Scalar value clamping operation.

        The clamp_scalar operation constrains all elements of a tensor to be within
        a specified range. This operation applies element-wise clamping to the
        input tensor, ensuring that all values fall within the range [min, max].
        Values less than min are set to min, and values greater than max are set
        to max. This is commonly used to ensure that tensor values stay within a
        valid range.

        Args:
            in0: The input tensor to clamp
            min_arg: The minimum value for clamping
            max_arg: The maximum value for clamping
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The clamped tensor

        Example:
            # Clamp values to the range [2.0, 5.0]
            # Input tensor: [[0, 1, 2, 3, 4, 5, 6, 7]]
            result = clamp_scalar(input_tensor, min_arg=2.0, max_arg=5.0)
            # Result: [[2, 2, 2, 3, 4, 5, 5, 5]]
            # Values < 2.0 are clamped to 2.0, values > 5.0 are clamped to 5.0
        """
        kwargs = {"min": min_arg, "max": max_arg}
        return self.op_proxy(
            torch.clamp,
            ttir.ClampScalarOp,
            [in0],
            ttir_kwargs=kwargs,
            golden_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def clamp_tensor(
        self,
        in0: Operand,
        in1: Operand,
        in2: Operand,
        in3: Operand,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Tensor value clamping operation.

        The clamp_tensor operation constrains elements of a tensor to be within
        ranges specified by min and max tensors. Unlike clamp_scalar, which uses
        scalar values for min and max, this operation uses tensor values for
        element-wise clamping. Each element in the input tensor is clamped
        between the corresponding elements in the min and max tensors. This
        allows for different clamping ranges for different elements.

        Args:
            in0: The input tensor to clamp
            in1: The tensor containing minimum values for clamping
            in2: The tensor containing maximum values for clamping
            in3: The output tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The clamped tensor

        Example:
            # Clamp values using min and max tensors
            # Input tensor: [[0, 1, 2, 3, 4, 5, 6, 7]]
            # Min tensor:   [[2, 2, 2, 3, 3, 3, 0, 0]]
            # Max tensor:   [[5, 5, 5, 9, 9, 9, 6, 6]]
            result = clamp_tensor(input_tensor, min_tensor, max_tensor, output_tensor)
            # Result: [[2, 2, 2, 3, 4, 5, 6, 6]]
            # Each element is clamped between its corresponding min and max values
        """
        return self.op_proxy(
            torch.clamp,
            ttir.ClampTensorOp,
            [in0, in1, in2, in3],
            golden_kwargs={
                "input": self._get_golden_tensor(in0),
                "min": self._get_golden_tensor(in1),
                "max": self._get_golden_tensor(in2),
                "out": self._get_golden_tensor(in3),
            },
            organize_ttir_args=lambda i, o, _: (
                self._get_type(o),
                i[0],
                i[1],
                i[2],
                i[3],
            ),
            organize_golden_args=lambda i: 0,
            unit_attrs=unit_attrs,
        )

    def zeros(
        self,
        shape: Shape,
        data_type: Optional[Type] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Creates a tensor filled with zeros.

        The zeros operation creates a tensor filled with zeros of the specified
        shape. This operation is commonly used to initialize tensors with zero
        values. It takes a shape attribute and produces a tensor of that shape
        with all elements set to zero.

        Args:
            shape: The shape of the tensor to create
            data_type: Optional type for the tensor elements. If None, uses default
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The tensor filled with zeros

        Example:
            # Create a 3D tensor of zeros with shape [64, 28, 28]
            result = zeros([64, 28, 28])
            # Result: A tensor of shape [64, 28, 28] filled with zeros

            # Create a 2D tensor of zeros with shape [3, 4]
            result = zeros([3, 4])
            # Result: [[0.0, 0.0, 0.0, 0.0],
            #          [0.0, 0.0, 0.0, 0.0],
            #          [0.0, 0.0, 0.0, 0.0]]

        Note:
            The element type of the result tensor is determined by data_type if
            specified, otherwise by the default type. This operation is useful
            for initializing tensors before filling them with computed values
            or as a starting point for accumulation operations.
        """
        output = self.ranked_tensor_type(shape)
        dtype = data_type if data_type is not None else self._default_dtype
        return self.op_proxy(
            torch.zeros,
            ttir.ZerosOp,
            [],
            golden_kwargs={"size": shape},
            ttir_kwargs={"result": output, "shape": shape},
            organize_ttir_args=lambda i, o, shape: 0,
            output_type=dtype,
            unit_attrs=unit_attrs,
        )

    def ones(self, shape: Shape, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Creates a tensor filled with ones.

        The ones operation creates a tensor filled with ones of the specified
        shape. This operation is commonly used to initialize tensors with one
        values. It takes a shape attribute and produces a tensor of that shape
        with all elements set to one.

        Args:
            shape: The shape of the tensor to create
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The tensor filled with ones

        Example:
            # Create a 3D tensor of ones with shape [64, 28, 28]
            result = ones([64, 28, 28])
            # Result: A tensor of shape [64, 28, 28] filled with ones

            # Create a 2D tensor of ones with shape [3, 4]
            result = ones([3, 4])
            # Result: [[1.0, 1.0, 1.0, 1.0],
            #          [1.0, 1.0, 1.0, 1.0],
            #          [1.0, 1.0, 1.0, 1.0]]

        Note:
            The element type of the result tensor is determined by the default
            type. This operation is useful for initializing tensors before
            scaling them or as a starting point for operations that require
            tensors filled with ones, such as creating masks or constant
            multipliers.
        """
        output = self.ranked_tensor_type(shape)
        return self.op_proxy(
            torch.ones,
            ttir.OnesOp,
            [],
            golden_kwargs={"size": shape},
            ttir_kwargs={"result": output, "shape": shape},
            organize_ttir_args=lambda i, o, shape: 0,
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def reverse(
        self, in0: Operand, dims: List[int], unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Tensor reversal operation.

        The reverse operation reverses the order of elements in the input tensor
        along the specified dimensions. This operation flips the elements of a
        tensor along one or more axes, which is useful for operations like
        sequence reversal, matrix transposition with reversal, and other tensor
        manipulations that require changing the order of elements.

        Args:
            in0: The input tensor to reverse
            dims: The dimensions along which to reverse the tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The reversed tensor

        Example:
            # Reverse a 3x2 tensor along dimension 1 (columns)
            # Input tensor:
            # [[1, 2],
            #  [3, 4],
            #  [5, 6]]
            result = reverse(input_tensor, dims=[1])
            # Result:
            # [[2, 1],
            #  [4, 3],
            #  [6, 5]]

            # Reverse a 3x2 tensor along both dimensions
            # Input tensor:
            # [[1, 2],
            #  [3, 4],
            #  [5, 6]]
            result = reverse(input_tensor, dims=[0, 1])
            # Result:
            # [[6, 5],
            #  [4, 3],
            #  [2, 1]]
        """
        return self.op_proxy(
            torch.flip,
            ttir.ReverseOp,
            [in0],
            golden_kwargs={"dims": dims},
            ttir_kwargs={"dimensions": dims},
            unit_attrs=unit_attrs,
        )

    def linear(
        self,
        in0: Operand,
        in1: Operand,
        bias: Optional[Operand] = None,
        transpose_a: bool = False,
        transpose_b: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Linear transformation operation.

        The `linear` operation performs a linear transformation by computing the
        matrix multiplication of tensors a and b with an optional addition of a
        bias tensor. This operation is commonly used in neural networks to
        implement fully connected layers. It computes the matrix multiplication
        of the input tensor with a weight tensor and adds an optional bias.

        Args:
            in0: The input tensor
            in1: The weight tensor
            bias: Optional bias tensor to add to the result of the matrix
                multiplication
            transpose_a: Whether to transpose tensor a before multiplication
            transpose_b: Whether to transpose tensor b before multiplication
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The result of the linear transformation

        Example:
            # Linear transformation with bias
            # Input tensor: batch_size=10, sequence_length=64, input_dim=32
            # Weight tensor: input_dim=32, output_dim=128
            # Bias tensor: output_dim=128
            result = linear(input_tensor, weight_tensor, bias_tensor)
            # Output shape: [10, 64, 128]

            # Linear transformation without bias
            result = linear(input_tensor, weight_tensor)
            # Output shape: [10, 64, 128]

        Note:
            The shapes of the tensors must be compatible for matrix
            multiplication. For a 3D input tensor with shape [batch_size,
            sequence_length, input_dim], the weight tensor should have shape
            [input_dim, output_dim], and the bias tensor should have shape
            [output_dim]. The resulting tensor will have shape [batch_size,
            sequence_length, output_dim].

            The operation computes: result = matmul(a, b) + bias
        """
        kwargs = {"transpose_a": transpose_a, "transpose_b": transpose_b, "bias": bias}
        return self.op_proxy(
            self.linear_golden_function,
            ttir.LinearOp,
            [in0, in1],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def linear_golden_function(
        self,
        a: Operand,
        b: Operand,
        bias: Optional[Operand] = None,
        transpose_a: bool = False,
        transpose_b: bool = False,
    ) -> OpView:
        a = torch.transpose(a, 0, 1) if transpose_a else a
        b = torch.transpose(b, 0, 1) if transpose_a else b
        output = torch.matmul(a, b)
        bias = (
            torch.zeros(list(output.shape))
            if not bias
            else self._get_golden_tensor(bias)
        )
        bias = (
            torch.broadcast_to(bias, list(output.shape))
            if bias.shape != output.shape
            else bias
        )
        return torch.add(output, bias)

    def matmul(
        self,
        in0: Operand,
        in1: Operand,
        bias: Optional[Operand] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Matrix multiplication operation.

        The `matmul` operation computes the matrix multiplication of two tensors.
        This operation performs matrix multiplication between tensors a and b.
        For 2D tensors, this computes the standard matrix product. For tensors
        with more dimensions, it applies batched matrix multiplication.

        Args:
            in0: The first input tensor
            in1: The second input tensor
            bias: Optional bias tensor to add to the result
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The result of the matrix multiplication

        Example:
            # Basic matrix multiplication of 2D tensors
            # Matrix A with shape [3,4]
            # Matrix B with shape [4,5]
            result = matmul(a, b)
            # Result: tensor with shape [3,5]

            # Batched matrix multiplication
            # Batch of 2 matrices A with shape [2,3,4]
            # Batch of 2 matrices B with shape [2,4,5]
            result = matmul(a, b)
            # Result: tensor with shape [2,3,5]

        Note:
            The inner dimensions of the input tensors must be compatible for
            matrix multiplication. If a has shape [..., m, k] and b has shape
            [..., k, n], then the result will have shape [..., m, n].
        """
        inputs = [in0, in1]
        if bias:
            inputs.append(bias)
        return self.op_proxy(
            torch.matmul,
            ttir.MatmulOp,
            inputs,
            unit_attrs=unit_attrs,
        )

    def permute(
        self,
        in0: Operand,
        in1: Operand,
        permutation: List[int],
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Tensor dimension permutation operation.

        The `permute` operation reorders the dimensions of the input tensor
        according to the specified permutation. This operation is similar to
        transpose but generalizes to tensors of any rank. It rearranges the
        dimensions of the input tensor based on the permutation attribute,
        which specifies the new order of dimensions.

        Args:
            in0: The input tensor to permute
            in1: The output tensor
            permutation: The permutation of the input tensor dimensions. This
                must be a valid permutation of the indices [0, 1, ..., rank-1]
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The permuted tensor

        Example:
            # Transpose a 2D tensor (swap dimensions 0 and 1)
            # Input tensor with shape [3,4]
            result = permute(input_tensor, output_tensor, permutation=[1, 0])
            # Result: tensor with shape [4,3], equivalent to transposing

            # Permute a 3D tensor
            # Input tensor with shape [2,3,4]
            result = permute(input_tensor, output_tensor, permutation=[1, 2, 0])
            # Result: tensor with shape [3,4,2]

        Note:
            The permutation must contain exactly one occurrence of each integer
            in the range [0, rank-1], where rank is the number of dimensions
            in the input tensor. The shape of the output tensor is determined
            by permuting the dimensions of the input tensor according to the
            permutation. For example, if the input shape is [2,3,4] and the
            permutation is [1,2,0], then the output shape will be [3,4,2].
        """
        return self.op_proxy(
            torch.permute,
            ttir.PermuteOp,
            [in0, in1],
            golden_kwargs={"dims": tuple(permutation)},
            ttir_kwargs={"permutation": DenseI64ArrayAttr.get(permutation)},
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            organize_ttir_args=lambda i, o, _: (self._get_type(i[1]), i[0], i[1]),
            unit_attrs=unit_attrs,
        )

    def upsample2d(
        self,
        in0: Operand,
        in1: Operand,
        scale_factor: Union[int, List[int]],
        mode: str = "nearest",
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Upsample 2D operation.

        The upsample2d operation increases the spatial dimensions (height and
        width) of an input tensor. This operation is commonly used in neural
        networks to increase the spatial resolution of feature maps. It
        supports different upsampling algorithms such as "nearest" and
        "bilinear" interpolation. The input tensor is assumed to be in NHWC
        format (batch, height, width, channels).

        Args:
            in0: The input tensor to upsample, in NHWC format
            in1: The output tensor
            scale_factor: The scale factor for upsampling in height and width
                dimensions. If a single integer is provided, it's used for
                both dimensions. If a list is provided, the first value is
                used for height and the second for width
            mode: The upsampling algorithm to use. Currently supported values
                are "nearest" for nearest neighbor interpolation and
                "bilinear" for bilinear interpolation
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The upsampled tensor

        Example:
            # Upsample with different scale factors for height and width
            # Input tensor: [batch=10, height=64, width=32, channels=3]
            result = upsample2d(input_tensor, output_tensor,
                              scale_factor=[2, 4], mode="bilinear")
            # Result: tensor with shape [10,128,128,3]

            # Upsample with the same scale factor for both dimensions
            # Input tensor: [1, 32, 32, 16]
            result = upsample2d(input_tensor, output_tensor,
                              scale_factor=2, mode="nearest")
            # Result: tensor with shape [1,64,64,16]

        Note:
            The output height is calculated as input_height * scale_factor[0]
            and the output width as input_width * scale_factor[1]. The batch
            and channel dimensions remain unchanged.
        """
        output_shape = self._get_golden_tensor(in1).shape
        kwargs = {
            "scale_factor": (
                IntegerAttr.get(IntegerType.get_signed(32), scale_factor)
                if isinstance(scale_factor, int)
                else DenseI32ArrayAttr.get(scale_factor)
            ),
            "mode": mode,
        }
        return self.op_proxy(
            self.upsample2d_golden_function,
            ttir.Upsample2dOp,
            [in0, in1],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(i[1]), i[0], o),
            output_shape=output_shape,
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def upsample2d_golden_function(
        self,
        in0: Operand,
        in1: Operand,
        scale_factor: Union[SI32Attr, DenseI32ArrayAttr],
        mode: str = "nearest",
    ) -> OpView:
        transposed_golden = torch.transpose(in0, 1, 3)
        golden_output_shape = in1.shape[1:-1]
        output = torch.nn.functional.interpolate(
            transposed_golden, size=golden_output_shape, mode=mode
        )
        return torch.transpose(output, 1, 3)

    def arange(
        self,
        result: Operand,
        start: int,
        end: int,
        step: int,
        arange_dimension: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Tensor range generation operation.

        The `arange` operation generates a tensor with evenly spaced values
        within a given interval. This operation creates a tensor with values
        from start to end (exclusive) with a step size of step, along the
        dimension specified by arange_dimension. It's similar to NumPy's
        arange function and is useful for creating tensors with regular
        sequences of values.

        Args:
            result: The output tensor
            start: The start value of the sequence
            end: The end value of the sequence (exclusive)
            step: The step size between values in the sequence
            arange_dimension: The dimension along which to generate the
                sequence
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The generated tensor containing the sequence

        Example:
            # Generate a 1D tensor with values [0, 1, 2, 3, 4]
            result = arange(output_tensor, start=0, end=5, step=1,
                          arange_dimension=0)

            # Generate a 2D tensor with the sequence along dimension 0
            # Result:
            # [[0, 0, 0],
            #  [1, 1, 1],
            #  [2, 2, 2],
            #  [3, 3, 3],
            #  [4, 4, 4]]
            result = arange(output_tensor, start=0, end=5, step=1,
                          arange_dimension=0)

            # Generate a 2D tensor with the sequence along dimension 1
            # Result:
            # [[0, 1, 2],
            #  [0, 1, 2],
            #  [0, 1, 2],
            #  [0, 1, 2],
            #  [0, 1, 2]]
            result = arange(output_tensor, start=0, end=3, step=1,
                          arange_dimension=1)
        """
        single_dim_tensor = torch.arange(
            start=start, end=end, step=step, dtype=self._get_golden_tensor(result).dtype
        )
        shape = self.get_shape(result)
        repeat_dims = []
        for i in range(len(shape)):
            if i == arange_dimension:
                repeat_dims.append(int(shape[i] / ((end - start) / step)))
            else:
                repeat_dims.append(shape[i])

        return self.op_proxy(
            torch.Tensor.repeat,
            ttir.ArangeOp,
            [result, single_dim_tensor],
            golden_kwargs={"repeats": tuple(repeat_dims)},
            ttir_kwargs={
                "start": start,
                "end": end,
                "step": step,
                "arange_dimension": arange_dimension,
            },
            organize_ttir_args=lambda i, o, _: (self._get_type(o),),
            organize_golden_args=lambda i: [i[1]],
            output_shape=shape,
            output_type=self.get_type_from_torch_dtype(
                self._get_golden_tensor(result).dtype
            ),
            unit_attrs=unit_attrs,
        )

    # TTIR top level generic ops
    # class TTIR_GenericElementwiseUnaryOp

    def exp(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise exponential operation.

        The `exp` operation computes the exponential of each element in the
        input tensor. For each element, it returns e^x, where e is the base
        of natural logarithms (approximately 2.71828).

        Args:
            in0: The input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The output tensor with exponential values

        Example:
            # Input tensor: [[1.0, 2.0, -3.0, 4.0], ...]
            result = exp(input_tensor)
            # Output: [[2.71828, 7.389056, 0.090031, 54.59815], ...]

        Mathematical definition: exp(x) = e^x
        """
        return self.eltwise_proxy(torch.exp, ttir.ExpOp, [in0], unit_attrs=unit_attrs)

    # class TTIR_GenericElementwiseBinaryOp

    def add(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Elementwise addition operation.

        The `add` operation performs an elementwise addition between two
        tensors. For each pair of corresponding elements, it adds the
        elements and places the result in the output tensor.

        Args:
            in0: The first input tensor
            in1: The second input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The output tensor with element-wise sums

        Example:
            # Integer tensors
            # in0: [10, 20, 30]
            # in1: [1, 2, 3]
            result = add(in0, in1)
            # Output: [11, 22, 33]

            # Float tensors
            # in0: [3.5, 0.0, -1.2]
            # in1: [1.5, 2.0, -3.2]
            result = add(in0, in1)
            # Output: [5.0, 2.0, -2.0]

        Note:
            The data type of the output tensor matches the data type of
            the input tensors.

        Mathematical definition: add(x, y) = x + y
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
        """Elementwise multiplication operation.

        The `multiply` operation performs an elementwise multiplication between
        two tensors. For each pair of corresponding elements, it multiplies
        the elements and places the result in the output tensor.

        Args:
            in0: The first input tensor
            in1: The second input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The output tensor with element-wise products

        Example:
            # Integer tensors
            # in0: [10, 20, 30]
            # in1: [1, 2, 3]
            result = multiply(in0, in1)
            # Output: [10, 40, 90]

            # Float tensors
            # in0: [3.5, 0.0, -1.2]
            # in1: [1.5, 2.0, -3.2]
            result = multiply(in0, in1)
            # Output: [5.25, 0.0, -3.84]

        Note:
            The data type of the output tensor matches the data type of
            the input tensors.

        Mathematical definition: multiply(x, y) = x * y
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
        """Elementwise subtract operation.

        The subtract operation performs an elementwise subtraction between
        two tensors. For each pair of corresponding elements, it subtracts
        the element in the second tensor from the element in the first
        tensor and places the result in the output tensor.

        Args:
            in0: The first input tensor (minuend)
            in1: The second input tensor (subtrahend)
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The output tensor with element-wise differences

        Example:
            # Integer tensors
            # in0: [10, 20, 30]
            # in1: [1, 2, 3]
            result = subtract(in0, in1)
            # Output: [9, 18, 27]

            # Float tensors
            # in0: [3.5, 0.0, -1.2]
            # in1: [1.5, 2.0, -3.2]
            result = subtract(in0, in1)
            # Output: [2.0, -2.0, 2.0]

        Note:
            The data type of the output tensor matches the data type of
            the input tensors.

        Mathematical definition: subtract(x, y) = x - y
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
        """Elementwise division operation.

        The `div` operation performs an elementwise division between
        two tensors. For each pair of corresponding elements, it divides
        the elements and places the result in the output tensor.

        Args:
            in0: The first input tensor
            in1: The second input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The output tensor with element-wise quotients

        Example:
            # Integer tensors
            # in0: [10, 20, 30]
            # in1: [1, 2, 3]
            result = div(in0, in1)
            # Output: [10, 10, 10]

            # Float tensors
            # in0: [3.5, 0.0, -1.2]
            # in1: [1.5, 2.0, -3.2]
            result = div(in0, in1)
            # Output: [2.33..., 0.0, 0.374999...]

        Note:
            The data type of the output tensor matches the data type of
            the input tensors.

        Mathematical definition: div(x, y) = x / y
        """
        return self.eltwise_proxy(
            torch.div, ttir.DivOp, [in0, in1], unit_attrs=unit_attrs
        )

    def maximum(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """Elementwise maximum operation.

        The `maximum` operation computes the elementwise maximum between two tensors.
        For each pair of corresponding elements, it selects the larger value and places
        it in the output tensor. This operation has the idempotence property, meaning
        that applying it twice with the same second operand returns the original result:
        maximum(maximum(x, y), y) = maximum(x, y).

        Note: When comparing with NaN values, NaN is typically not selected as the maximum value.

        Args:
            in0: First input tensor
            in1: Second input tensor
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: A tensor containing the elementwise maximum of the inputs

        Example:
            # Minimum operation
            # Input tensors:
            # lhs: [[3, 2, 7], [1, 4, 4]]
            # rhs: [[1, 4, 2], [1, 2, 3]]
            # Output tensor: [[3, 4, 7], [1, 4, 4]]
            result = maximum(lhs, rhs)

            # Example with floating point values
            # Input tensors:
            # float_lhs: [3.5, -2.1, 0.0]
            # float_rhs: [1.2, -5.0, 0.0]
            # Output tensor: [3.5, -2.1, 0.0]
            result = maximum(float_lhs, float_rhs)
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
        """Quantize operation.

        The `quantize` operation converts a tensor into a quantized tensor using the specified
        scale and zero_point parameters. For each element in the input tensor, the quantization
        is computed as:
            output[i] = (input[i] / scale) + zero_point

        Args:
            in0: Input tensor to be quantized
            scale: The scale factor for quantization
            zero_point: The zero point offset value
            dtype: The target data type for quantization
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The quantized tensor

        Example:
            # Quantize a float32 tensor to int8
            # input: tensor with shape [64, 128] and dtype float32
            # scale: 0.1 (each step represents 0.1 in the original scale)
            # zero_point: 128 (value 128 in quantized space represents 0.0)
            result = quantize(input, scale=0.1, zero_point=128, dtype=torch.int8)
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
        """Dequantize operation.

        The `dequantize` operation converts a quantized tensor back into a floating-point tensor
        using the specified scale and zero_point parameters. For each element in the input
        tensor, the dequantization is computed as:
            output[i] = (input[i] - zero_point) * scale

        Args:
            in0: Input quantized tensor to be dequantized
            scale: The scale factor used in the original quantization
            zero_point: The zero point offset value used in the original quantization
            dtype: The target floating-point data type
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The dequantized floating-point tensor

        Example:
            # Dequantize an int8 tensor back to float32
            # input: tensor with shape [64, 128] and dtype int8
            # scale: 0.1 (each step represents 0.1 in the original scale)
            # zero_point: 128 (value 128 in quantized space represents 0.0)
            result = dequantize(input, scale=0.1, zero_point=128, dtype=torch.float32)
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
        """Requantize operation.

        The `requantize` operation converts a quantized tensor from one scale and zero-point
        to another. For each element in the input tensor, the requantization is computed as:
            output[i] = round((input[i] - input_zero_point) * (input_scale / output_scale)) + output_zero_point

        Args:
            in0: Input quantized tensor to be requantized
            scale: The new scale factor for requantization
            zero_point: The new zero point offset value
            dtype: The target quantized data type
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: The requantized tensor with new scale and zero point

        Example:
            # Requantize a tensor from one quantization scheme to another
            # input: tensor with shape [64, 128] and dtype int8, scale=0.1, zero_point=128
            # New parameters: scale=0.2, zero_point=100
            result = requantize(input, scale=0.2, zero_point=100, dtype=torch.int8)
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
        """Layout transition operation.

        The `to_layout` operation transitions tensors from one layout to another. This can include:
        - Transitioning between different memory spaces (e.g., DRAM to L1)
        - Transitioning between different data types (e.g., f32 to f16)
        - Transitioning between different tile sizes (e.g., 1x16 to 32x32)
        - Transitioning between different tensor sharding
        - Some combination of the above

        Args:
            in0: Input tensor to be transformed
            output_type: The target RankedTensorType specifying the desired layout
            unit_attrs: Optional list of unit attributes
            **kwargs: Additional keyword arguments for the layout transformation

        Returns:
            OpView: The tensor with the transformed layout

        Example:
            # Transform a tensor from system memory to L1 memory
            # layout = metal_layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #system>>
            # layout1 = metal_layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #l1_>>
            result = to_layout(input_tensor, output_type=layout1_type)
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
        """All Gather Operation

        The `all_gather` operation performs a typical all gather over the devices on a system.

        Args:
            input: Input tensor to be gathered
            all_gather_dim: The dimension over which to gather w.r.t the tensor dimensions
            cluster_axis: Either `0` or `1`. This determines which direction
                the gather takes place in w.r.t the shape of the mesh. For
                example, if there was a mesh shape of [2,4] (i.e. device ids
                are: [[0, 1, 2, 3], [4, 5, 6, 7]] and `cluster_axis` is `0`,
                then the `0`th dimension will be used to gather, and there will
                be 4 separate gathers taking place (i.e. (0, 4), (1, 5), (2,
                6), & (3, 7)). If `cluster_axis` is instead set to `1`, the
                first dimension will be used and there will be two gathers
                (i.e. (0, 1, 2, 3) & (4, 5, 6, 7)).

        Returns:
            OpView: The tensor expanded in the `all_gather_dim` dimension with gathered data.

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
        """All Reduce Operation

        The `all_reduce` operation performs a typical all reduce operation,
        reducing `reduce_type` over other devices on the system and gathering
        the results in the result of this op.

        Args:
            input: Input tensor to be reduced
            reduce_type: The type of reduction to perform
            cluster_axis: Either `0` or `1`. This determines which direction
                the reduction takes place in w.r.t the shape of the mesh. For
                example, if there was a mesh shape of [2,4] (i.e. device ids
                are: [[0, 1, 2, 3], [4, 5, 6, 7]] and `cluster_axis` is `0`,
                then the `0`th dimension will be used to reduce, and there will
                be 4 separate reductions taking place (i.e. (0, 4), (1, 5), (2,
                6), & (3, 7)). If `cluster_axis` is instead set to `1`, the
                first dimension will be used and there will be two reductions
                (i.e. (0, 1, 2, 3) & (4, 5, 6, 7)).

        Returns:
            OpView: The result of the reduction. This will be the same across
            all devices that execute this `all_reduce`.

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
        """Reduce Scatter Operation

        The `reduce_scatter` operation performs a typical reduce scatter over
        devices on the system, using `reduce_type` as reduction function. The
        results are then scattered back across the devices according to
        `scatter_dim`

        Args:
            input: Input tensor to be reduced
            reduce_type: The type of reduction to perform
            scatter_dim: The dimension over which to scatter w.r.t the tensor
            dimensions once the reduction is performed
            cluster_axis: Either `0` or `1`. This determines which direction
                the reduction takes place in w.r.t the shape of the mesh. For
                example, if there was a mesh shape of [2,4] (i.e. device ids
                are: [[0, 1, 2, 3], [4, 5, 6, 7]] and `cluster_axis` is `0`,
                then the `0`th dimension will be used to reduce, and there will
                be 4 separate reductions taking place (i.e. (0, 4), (1, 5), (2,
                6), & (3, 7)). If `cluster_axis` is instead set to `1`, the
                first dimension will be used and there will be two reductions
                (i.e. (0, 1, 2, 3) & (4, 5, 6, 7)).

        Returns:
            OpView: The result of the reduction. This result will be different
            across all the devices according to the scatter

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
        """Collective Permute Operation

        This operation ingests a multi-device tensor spread across
        multi-devices and will shuffle the data according to
        source_target_pairs [['src', 'dest']].

        Args:
            input: The input tensor to be permuted
            source_target_pairs: List of pairs of source and target device ids

        Example:
            For a 1x2 mesh, the following will take the device shard living in
            device 0 and move it to device 1. The device shard living in device
            1 will move to device 0. %source_target_pairs: [[0, 1], [1, 0]]

            In the case of missing 'dest', the device shard living on that
            device will contain values of 0. For example, device shard living
            in device 0 will contain 0 values. %source_target_pairs: [[0, 1]]

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
