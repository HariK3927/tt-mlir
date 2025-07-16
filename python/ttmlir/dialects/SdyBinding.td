// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef PYTHON_BINDINGS_TTMLIR_SDYOPS
#define PYTHON_BINDINGS_TTMLIR_SDYOPS

//include "/opt/ttmlir-toolchain/src/shardy/shardy/dialect/sdy/ir/ops.td"
//include "build/shardy/shardy/dialect/sdy/ir/ops.cc.inc"
include "shardy/dialect/sdy/ir/dialect.td"
include "shardy/dialect/sdy/ir/ops.td"
include "shardy/dialect/sdy/ir/op_interface.td"
//include "shardy/dialect/sdy/ir/op_interface.td"
src/shardy/shardy/integrations/python/ir/sdy_ops.td

#endif
