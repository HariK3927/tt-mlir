// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/L1InterleavedAnalysis.h"

#include "ttmlir/Dialect/TTNN/Analysis/MemReconfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"

namespace mlir::tt::ttnn {

void L1InterleavedAnalysis::analysisImplementation() {
  constexpr float tensorL1UsageCap = 0.8;
  const uint64_t availableL1CacheSize = tensorL1UsageCap * analysisInput.usableL1CacheSize;
  
  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
               "L1InterleavedAnalysis: Available L1 cache size: {}", 
               availableL1CacheSize);

  // Process each function's schedule
  for (const auto &[funcOp, schedule] : analysisInput.schedule) {
    uint64_t currentL1Usage = 0;
    
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                 "L1InterleavedAnalysis: Processing function {} with {} operations",
                 funcOp.getName(), schedule.size());

    // Go through schedule in order, trying to upgrade DRAM ops to L1 interleaved
    for (size_t i = 0; i < schedule.size(); ++i) {
      Operation *op = schedule[i];
      
      // Skip if operation doesn't use DRAM layout
      if (!usesDRAMLayout(op)) {
        continue;
      }
      
      // Skip if operation doesn't have L1 interleaved layout available
      if (!hasL1InterleavedLayout(op)) {
        continue;
      }
      
      // Check if upgrade is safe and beneficial
      if (!canUpgradeToL1Interleaved(op, schedule, i, currentL1Usage)) {
        continue;
      }
      
      // Get the L1 interleaved layout
      TTNNLayoutAttr l1Layout = getL1InterleavedLayout(op);
      uint64_t outputL1Usage = calculateOutputL1Usage(op, l1Layout);
      
      // Check if we have enough L1 memory
      if (currentL1Usage + outputL1Usage > availableL1CacheSize) {
        TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                     "L1InterleavedAnalysis: Skipping {} - insufficient L1 memory ({} + {} > {})",
                     op->getName(), currentL1Usage, outputL1Usage, availableL1CacheSize);
        continue;
      }
      
      // Create the upgraded config
      OpConfig upgradedConfig;
      upgradedConfig.outputLayout = l1Layout;
      
      // Find existing config with same op-specific attributes but DRAM layout
      const auto &legalConfigs = analysisInput.legalConfigs.at(op);
      for (const OpConfig &config : legalConfigs) {
        if (config.outputLayout.hasDRAMBufferType()) {
          upgradedConfig.opSpecificAttrs = config.opSpecificAttrs;
          break;
        }
      }
      
      // Record the upgrade
      analysisResult.upgradedConfigs[op] = upgradedConfig;
      currentL1Usage += outputL1Usage;
      
      TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                   "L1InterleavedAnalysis: Upgraded {} to L1 interleaved (L1 usage: {})",
                   op->getName(), currentL1Usage);
      
      // Insert memory reconfig if producer is still in DRAM
      for (auto operand : op->getOperands()) {
        Operation *producer = operand.getDefiningOp();
        if (!producer) continue;
        
        // Check if producer uses DRAM and wasn't upgraded
        if (usesDRAMLayout(producer) && 
            analysisResult.upgradedConfigs.find(producer) == analysisResult.upgradedConfigs.end()) {
          
          Edge edge(producer, op, 0); // Simplified - using operand index 0
          
          // Create memory reconfig entry to convert DRAM -> L1 interleaved
          MemReconfigEntry entry;
          entry.srcLayout = analysisInput.legalConfigs.at(producer)[0].outputLayout; // DRAM layout
          entry.dstLayout = l1Layout.withBufferType(BufferType::L1); // L1 interleaved input
          entry.edge = edge;
          
          analysisResult.memReconfigEntryMap[edge] = entry;
          
          TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
                       "L1InterleavedAnalysis: Added memory reconfig for edge {} -> {}",
                       producer->getName(), op->getName());
        }
      }
    }
  }
  
  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer,
               "L1InterleavedAnalysis: Completed - upgraded {} operations",
               analysisResult.upgradedConfigs.size());
}

bool L1InterleavedAnalysis::hasL1InterleavedLayout(Operation *op) const {
  const auto it = analysisInput.legalConfigs.find(op);
  if (it == analysisInput.legalConfigs.end()) {
    return false;
  }
  
  for (const OpConfig &config : it->second) {
    if (config.outputLayout.hasInterleavedL1TensorMemoryLayout()) {
      return true;
    }
  }
  return false;
}

TTNNLayoutAttr L1InterleavedAnalysis::getL1InterleavedLayout(Operation *op) const {
  const auto it = analysisInput.legalConfigs.find(op);
  assert(it != analysisInput.legalConfigs.end());
  
  for (const OpConfig &config : it->second) {
    if (config.outputLayout.hasInterleavedL1TensorMemoryLayout()) {
      return config.outputLayout;
    }
  }
  
  llvm_unreachable("Operation should have L1 interleaved layout");
}

bool L1InterleavedAnalysis::usesDRAMLayout(Operation *op) const {
  const auto it = analysisInput.legalConfigs.find(op);
  if (it == analysisInput.legalConfigs.end()) {
    return false;
  }
  
  // Check if the current/selected config uses DRAM
  // For simplicity, assume first config is the current one
  if (!it->second.empty()) {
    return it->second[0].outputLayout.hasDRAMBufferType();
  }
  return false;
}

uint64_t L1InterleavedAnalysis::calculateOutputL1Usage(Operation *op, const TTNNLayoutAttr &layout) const {
  // Get the output tensor type
  auto outputType = mlir::cast<RankedTensorType>(op->getResult(0).getType());
  
  // Calculate tensor size in bytes
  uint64_t elementSizeBytes = layout.getScalarElementType().getIntOrFloatBitWidth() / 8;
  uint64_t numElements = 1;
  for (int64_t dim : outputType.getShape()) {
    numElements *= dim;
  }
  
  // For L1 interleaved, we need to account for tile padding and memory layout
  // This is a simplified calculation - real implementation would use OpModel
  constexpr uint64_t tileSize = 32 * 32; // 32x32 tile
  uint64_t numTiles = (numElements + tileSize - 1) / tileSize;
  
  return numTiles * tileSize * elementSizeBytes;
}

bool L1InterleavedAnalysis::canUpgradeToL1Interleaved(Operation *op, 
                                                       const llvm::SmallVector<Operation *> &schedule,
                                                       size_t opIndex,
                                                       uint64_t &currentL1Usage) const {
  // Check if operation has exactly one user that is immediate next in schedule
  return hasImmediateConsumer(op, schedule, opIndex);
}

bool L1InterleavedAnalysis::hasImmediateConsumer(Operation *op, 
                                                 const llvm::SmallVector<Operation *> &schedule,
                                                 size_t opIndex) const {
  // Check if operation has exactly one user
  if (!op->hasOneUse()) {
    return false;
  }
  
  // Get the single user
  Operation *user = *op->getUsers().begin();
  
  // Check if the user is the immediate next operation in schedule
  if (opIndex + 1 >= schedule.size()) {
    return false;
  }
  
  return schedule[opIndex + 1] == user;
}

} // namespace mlir::tt::ttnn