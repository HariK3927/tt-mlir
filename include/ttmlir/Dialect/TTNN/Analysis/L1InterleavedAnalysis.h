// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_L1INTERLEAVEDANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_L1INTERLEAVEDANALYSIS_H

#include "ttmlir/Dialect/TTNN/Analysis/Edge.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemReconfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"

#include <vector>

namespace mlir::tt::ttnn {

struct L1InterleavedAnalysisInput {
  llvm::DenseMap<Operation *, std::vector<OpConfig>> legalConfigs;
  llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> schedule;
  unsigned usableL1CacheSize = 0;

  L1InterleavedAnalysisInput() : legalConfigs(), schedule() {}

  L1InterleavedAnalysisInput(
      const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
      const llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> &schedule,
      unsigned usableL1CacheSize)
      : legalConfigs(legalConfigs), schedule(schedule), 
        usableL1CacheSize(usableL1CacheSize) {}

  bool operator==(const L1InterleavedAnalysisInput &rhs) const {
    return legalConfigs == rhs.legalConfigs && schedule == rhs.schedule;
  }

  bool operator!=(const L1InterleavedAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

struct L1InterleavedAnalysisResult {
  llvm::DenseMap<Operation *, OpConfig> upgradedConfigs;
  llvm::DenseMap<Edge, MemReconfigEntry> memReconfigEntryMap;

  L1InterleavedAnalysisResult() : upgradedConfigs(), memReconfigEntryMap() {}

  L1InterleavedAnalysisResult(
      const llvm::DenseMap<Operation *, OpConfig> &upgradedConfigs,
      const llvm::DenseMap<Edge, MemReconfigEntry> &memReconfigEntryMap)
      : upgradedConfigs(upgradedConfigs), memReconfigEntryMap(memReconfigEntryMap) {}
};

// Analysis that runs after spillToDRAM to try upgrading DRAM operations
// to L1 interleaved when:
// 1. The operation has exactly one user
// 2. That user is the immediate next operation in the schedule
// 3. L1 memory constraints are satisfied
//
class L1InterleavedAnalysis : public TTNNAnalysis<L1InterleavedAnalysisInput,
                                                  L1InterleavedAnalysisResult> {

private:
  void analysisImplementation() override;
  bool applyOverrides() override { return false; }

  // Check if operation has L1 interleaved layout available
  bool hasL1InterleavedLayout(Operation *op) const;
  
  // Get L1 interleaved layout for operation
  TTNNLayoutAttr getL1InterleavedLayout(Operation *op) const;
  
  // Check if operation currently uses DRAM layout
  bool usesDRAMLayout(Operation *op) const;
  
  // Calculate L1 memory usage for operation's output
  uint64_t calculateOutputL1Usage(Operation *op, const TTNNLayoutAttr &layout) const;
  
  // Check if upgrading operation to L1 interleaved is safe and beneficial
  bool canUpgradeToL1Interleaved(Operation *op, 
                                 const llvm::SmallVector<Operation *> &schedule,
                                 size_t opIndex,
                                 uint64_t &currentL1Usage) const;

  // Check if operation has exactly one user that is immediate next in schedule
  bool hasImmediateConsumer(Operation *op, 
                           const llvm::SmallVector<Operation *> &schedule,
                           size_t opIndex) const;

public:
  L1InterleavedAnalysis(Operation *op) : TTNNAnalysis(op) {}
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_L1INTERLEAVEDANALYSIS_H
