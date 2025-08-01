// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "executor.h"
#include "executor_utils.h"

#include "tools/profiler/op_profiler.hpp"
#include "tracy/Tracy.hpp"
#include "tt/runtime/debug.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/dylib.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttmetal/profiler.h"
#include "tt/runtime/detail/ttmetal/ttmetal.h"
#include "tt/runtime/perf.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"
#include "tt/runtime/workarounds.h"

#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Target/TTMetal/types_generated.h"
#include "ttmlir/Version.h"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace tt::runtime::ttmetal {

namespace target = ::tt::target;
namespace tt_metal = ::tt::tt_metal;
namespace distributed = ::tt::tt_metal::distributed;

namespace {
class MCQExecutor {
public:
  MCQExecutor(
      distributed::MeshDevice *meshDevice,
      const flatbuffers::Vector<
          flatbuffers::Offset<tt::target::metal::BufferRef>> *programInputs,
      const std::vector<Tensor> &inputs, common::DylibManager &&dylibManager,
      bool blockingCQ);

  const std::vector<Tensor> &getOutputs() const { return outputs; }

  void execute(const target::metal::CommandQueue *commandQueue);

private:
  void execute(const target::metal::Command *command);
  void execute(const target::metal::HostAllocCommand *command);
  void execute(const target::metal::ReturnCommand *command);
  void execute(const target::metal::EnqueueProgramCommand *command,
               const char *loc, const char *debugInfo);
  void execute(const target::metal::EnqueueWriteBufferCommand *command);
  void execute(const target::metal::EnqueueReadBufferCommand *command);
  void execute(const target::metal::CreateBufferCommand *command);
  void execute(const target::metal::DeallocateBufferCommand *command);
  void execute(const target::metal::CreateEventCommand *command);
  void execute(const target::metal::EnqueueRecordEventCommand *command);
  void execute(const target::metal::EnqueueWaitForEventCommand *command);
  void execute(const target::metal::EventSynchronizeCommand *command);
  void execute(const target::metal::EventQueryCommand *command);
  void execute(const target::metal::MemrefCopyCommand *command);
  void execute(const target::metal::CpuCommand *command);
  void execute(const target::metal::FinishCommand *command);

  std::uint64_t getUniqueProgramRuntimeId() { return nextProgramRuntimeId++; }

private:
  distributed::MeshDevice *meshDevice;
  std::vector<std::shared_ptr<distributed::MeshEvent>> initMeshEvents;
  std::unordered_map<std::uint32_t, std::shared_ptr<distributed::MeshBuffer>>
      meshBuffers;
  std::unordered_map<std::uint32_t, Tensor> hostBuffers;
  std::unordered_map<std::uint32_t, std::shared_ptr<distributed::MeshEvent>>
      meshEvents;
  std::vector<Tensor> outputs;
  distributed::MeshCommandQueue *mcq;
  bool blockingCQ;
  const char *currentProgramName;
  DeviceAddressValidator deviceAddressValidator;
  common::DylibManager dylibManager;
  std::uint64_t nextProgramRuntimeId = 10000; // Start at a greppable number.
};
} // namespace

MCQExecutor::MCQExecutor(
    distributed::MeshDevice *meshDevice,
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::metal::BufferRef>>
        *programInputs,
    const std::vector<Tensor> &inputs, common::DylibManager &&dylibManager,
    bool blockingCQ)
    : meshDevice(meshDevice), blockingCQ(blockingCQ),
      deviceAddressValidator(meshDevice->get_devices().at(0)),
      dylibManager(std::move(dylibManager)) {
  initMeshEvents.reserve(inputs.size());

  std::uint32_t inputIndex = 0;
  for (const Tensor &input : inputs) {
    const target::metal::BufferRef *ref = programInputs->Get(inputIndex++);
    std::visit(utils::overloaded{
                   [&](const TensorDesc &) {
                     auto [_, inserted] =
                         hostBuffers.try_emplace(ref->global_id(), input);
                     LOG_ASSERT(inserted);
                   },
                   [&](const MeshBuffer &mesh_buffer) {
                     auto [_, inserted] =
                         meshBuffers.try_emplace(ref->global_id(), mesh_buffer);
                     LOG_ASSERT(inserted);
                   },
               },
               input.as<MetalTensor>(DeviceRuntime::TTMetal));

    auto meshEvent =
        input.event.asSharedPtr<distributed::MeshEvent>(DeviceRuntime::TTMetal);
    if (meshEvent) {
      initMeshEvents.push_back(meshEvent);
    }
  }
}

void MCQExecutor::execute(const target::metal::CommandQueue *commandQueue) {
  currentProgramName = commandQueue->name()->c_str();
  mcq = &meshDevice->mesh_command_queue(commandQueue->queue_id());

  for (const auto &mesh_event : initMeshEvents) {
    distributed::EventSynchronize(*mesh_event);
  }
  initMeshEvents.clear();

  for (const target::metal::Command *command : *commandQueue->commands()) {
    LOG_TRACE(logger::LogRuntimeTTMetalCommand,
              "Executing command: ", EnumNameCommandType(command->type_type()),
              "\n\t", command->debug_info()->c_str());
    execute(command);
  }
}

void MCQExecutor::execute(const target::metal::Command *command) {
  switch (command->type_type()) {
  case target::metal::CommandType::HostAllocCommand: {
    execute(command->type_as_HostAllocCommand());
    break;
  }
  case target::metal::CommandType::ReturnCommand: {
    execute(command->type_as_ReturnCommand());
    break;
  }
  case target::metal::CommandType::EnqueueProgramCommand: {
    execute(command->type_as_EnqueueProgramCommand(), command->loc()->c_str(),
            command->debug_info()->c_str());
    break;
  }
  case target::metal::CommandType::EnqueueWriteBufferCommand: {
    execute(command->type_as_EnqueueWriteBufferCommand());
    break;
  }
  case target::metal::CommandType::EnqueueReadBufferCommand: {
    execute(command->type_as_EnqueueReadBufferCommand());
    break;
  }
  case target::metal::CommandType::CreateBufferCommand: {
    execute(command->type_as_CreateBufferCommand());
    break;
  }
  case target::metal::CommandType::DeallocateBufferCommand: {
    execute(command->type_as_DeallocateBufferCommand());
    break;
  }
  case target::metal::CommandType::CreateEventCommand: {
    execute(command->type_as_CreateEventCommand());
    break;
  }
  case target::metal::CommandType::EnqueueRecordEventCommand: {
    execute(command->type_as_EnqueueRecordEventCommand());
    break;
  }
  case target::metal::CommandType::EnqueueWaitForEventCommand: {
    execute(command->type_as_EnqueueWaitForEventCommand());
    break;
  }
  case target::metal::CommandType::EventSynchronizeCommand: {
    execute(command->type_as_EventSynchronizeCommand());
    break;
  }
  case target::metal::CommandType::EventQueryCommand: {
    execute(command->type_as_EventQueryCommand());
    break;
  }
  case target::metal::CommandType::MemrefCopyCommand: {
    execute(command->type_as_MemrefCopyCommand());
    break;
  }
  case target::metal::CommandType::CpuCommand: {
    execute(command->type_as_CpuCommand());
    break;
  }
  case target::metal::CommandType::FinishCommand: {
    execute(command->type_as_FinishCommand());
    break;
  }
  case target::metal::CommandType::NONE: {
    LOG_FATAL("Unsupported CommandType::NONE");
    break;
  }
  }
}

void MCQExecutor::execute(const target::metal::HostAllocCommand *command) {
  LOG_ASSERT(command->dst()->address() == 0);
  const auto *bufferDesc = command->dst()->desc();
  LOG_ASSERT(bufferDesc->sharded_buffer_config() == nullptr);
  LOG_ASSERT(bufferDesc->shape()->size() > 0);

  std::vector<std::uint32_t> shape(bufferDesc->shape()->begin(),
                                   bufferDesc->shape()->end());
  TensorDesc desc(shape, bufferDesc->data_type(),
                  utils::tileAlignment(bufferDesc->data_type()));
  size_t size = desc.sizeBytes();
  auto data = std::shared_ptr<void>(std::malloc(size), std::free);
  if (!data) {
    LOG_FATAL("HostAllocCommand: Failed to allocate host memory.");
  }

  if (command->data() != nullptr) {
    assert(command->data()->size() == size);
    std::memcpy(data.get(), command->data()->data(), size);
  }

  std::shared_ptr<MetalTensor> tensor = std::make_shared<MetalTensor>(desc);
  auto [_, inserted] = hostBuffers.try_emplace(
      command->dst()->global_id(), std::static_pointer_cast<void>(tensor), data,
      DeviceRuntime::TTMetal);
  LOG_ASSERT(inserted);
}

void MCQExecutor::execute(const target::metal::ReturnCommand *command) {
  std::shared_ptr<distributed::MeshEvent> meshEvent = nullptr;
  if (workaround::Env::get().d2mReturnEvent) {
    meshEvent = std::make_shared<distributed::MeshEvent>(
        distributed::EnqueueRecordEventToHost(*mcq));
  } else {
    distributed::Finish(*mcq);
  }

  LOG_ASSERT(outputs.empty(),
             "Unexpected outputs, multiple returns not supported");
  outputs.reserve(command->results()->size());
  for (const auto *result : *command->results()) {
    auto meshBufferIter = meshBuffers.find(result->global_id());
    bool meshBufferFound = meshBufferIter != meshBuffers.end();
    auto hostBufferIter = hostBuffers.find(result->global_id());
    bool hostBufferFound = hostBufferIter != hostBuffers.end();
    LOG_ASSERT(meshBufferFound != hostBufferFound);
    if (meshBufferFound) {
      outputs.emplace_back(
          std::static_pointer_cast<void>(meshBufferIter->second), nullptr,
          std::static_pointer_cast<void>(meshEvent), DeviceRuntime::TTMetal);
    } else {
      outputs.emplace_back(hostBufferIter->second);
      outputs.back().event = Event(std::static_pointer_cast<void>(meshEvent),
                                   DeviceRuntime::TTMetal);
    }
  }
}

void MCQExecutor::execute(const target::metal::EnqueueProgramCommand *command,
                          const char *loc, const char *debugInfo) {
  ZoneScopedN("EnqueueProgramCommand");
  tt_metal::Program program = tt_metal::CreateProgram();
  program.set_runtime_id(getUniqueProgramRuntimeId());

  for (const target::metal::KernelConfig *kernelConfig :
       *command->program()->kernels()) {
    const target::metal::KernelSource *kernelSource =
        kernelConfig->kernel_as_KernelSource();
    LOG_ASSERT(kernelSource, "Only source kernels supported for now");
    std::string kernelSourceString(kernelSource->source()->c_str(),
                                   kernelSource->source()->size());

    CoreRangeSet coreRangeSet =
        common::toCoreRangeSet(kernelConfig->core_range_set());

    auto createSemaphore = [&](std::uint32_t initialValue,
                               CoreType coreType) -> std::uint32_t {
      return tt_metal::CreateSemaphore(program, coreRangeSet, initialValue,
                                       coreType);
    };

    tt_metal::KernelHandle handle = createKernel(
        program, kernelSourceString, coreRangeSet,
        createKernelConfig(kernelConfig, command->buffers(), meshBuffers,
                           command->cbs(), deviceAddressValidator,
                           createSemaphore),
        currentProgramName, debugInfo, kernelConfig->debug_info()->c_str());

    std::vector<uint32_t> rtArgsVec = processRuntimeArgs(
        kernelConfig->args()->rt_args(), command->buffers(), meshBuffers,
        command->cbs(), deviceAddressValidator, createSemaphore);
    tt_metal::SetRuntimeArgs(program, handle, coreRangeSet, rtArgsVec);
  }

  for (const target::metal::CBRef *cbRef : *command->cbs()) {
    CoreRangeSet coreRangeSet =
        common::toCoreRangeSet(cbRef->buffer_ref()
                                   ->desc()
                                   ->circular_buffer_config()
                                   ->core_range_set());
    tt_metal::CircularBufferConfig config =
        createCircularBufferConfig(cbRef, meshBuffers);
    tt_metal::CreateCircularBuffer(program, coreRangeSet, config);
  }

  auto meshWorkload = distributed::CreateMeshWorkload();
  auto deviceRange = distributed::MeshCoordinateRange(meshDevice->shape());

  distributed::AddProgramToMeshWorkload(meshWorkload, std::move(program),
                                        deviceRange);
  distributed::EnqueueMeshWorkload(*mcq, meshWorkload, blockingCQ);

  if (perf::Env::get().enablePerfTrace) {
    profiler::profileProgram(meshDevice, program, loc);
  }
}

void MCQExecutor::execute(
    const target::metal::EnqueueWriteBufferCommand *command) {
  ZoneScopedN("EnqueueWriteBufferCommand");

  void *src = hostBuffers.at(command->src()->global_id()).data.get();
  LOG_ASSERT(src);
  auto meshBuffer = meshBuffers.at(command->dst()->global_id());
  mcq->enqueue_write_mesh_buffer(meshBuffer, src, blockingCQ);
}

void MCQExecutor::execute(
    const target::metal::EnqueueReadBufferCommand *command) {
  ZoneScopedN("EnqueueReadBufferCommand");

  void *dst = hostBuffers.at(command->dst()->global_id()).data.get();
  LOG_ASSERT(dst);
  auto meshBuffer = meshBuffers.at(command->src()->global_id());
  mcq->enqueue_read_mesh_buffer(dst, meshBuffer, true);
}

void MCQExecutor::execute(const target::metal::CreateBufferCommand *command) {
  ZoneScopedN("CreateBufferCommand");
  if (meshBuffers.find(command->ref()->global_id()) == meshBuffers.end()) {
    meshBuffers[command->ref()->global_id()] = createMeshBufferFromBufferRef(
        meshDevice, command->ref(), deviceAddressValidator);
  }
}

void MCQExecutor::execute(
    const target::metal::DeallocateBufferCommand *command) {
  ZoneScopedN("DeallocateBufferCommand");
  auto meshBufferIter = meshBuffers.find(command->ref()->global_id());
  LOG_ASSERT(meshBufferIter != meshBuffers.end(), "Buffer not allocated");
  LOG_ASSERT(meshBufferIter->second != nullptr, "Buffer already deallocated");
  auto meshBuffer = meshBufferIter->second;
  meshBuffer->deallocate();
  meshBuffers.erase(meshBufferIter);
}

void MCQExecutor::execute(const target::metal::CreateEventCommand *command) {
  ZoneScopedN("CreateEventCommand");
  LOG_ASSERT(!meshEvents.contains(command->ref()->global_id()));
  // TODO(wooseoklee): CreateEventCommand should be updated once we confirm the
  // use cases of MeshEvent.
}

void MCQExecutor::execute(
    const target::metal::EnqueueRecordEventCommand *command) {
  ZoneScopedN("EnqueueRecordEventCommand");
  meshEvents[command->ref()->global_id()] =
      std::make_shared<distributed::MeshEvent>(
          distributed::EnqueueRecordEvent(*mcq));
}

void MCQExecutor::execute(
    const target::metal::EnqueueWaitForEventCommand *command) {
  ZoneScopedN("EnqueueWaitForEventCommand");
  auto mesh_event = meshEvents.at(command->ref()->global_id());
  distributed::EnqueueWaitForEvent(*mcq, *mesh_event);
}

void MCQExecutor::execute(
    const target::metal::EventSynchronizeCommand *command) {
  ZoneScopedN("EventSynchronizeCommand");
  auto mesh_event = meshEvents.at(command->ref()->global_id());
  distributed::EventSynchronize(*mesh_event);
}

void MCQExecutor::execute(const target::metal::EventQueryCommand *command) {
  ZoneScopedN("EventQueryCommand");
  auto mesh_event = meshEvents.at(command->ref()->global_id());
  // TODO(nsmith): Need flatbuffer support for tracking and doing something
  // with the result
  (void)distributed::EventQuery(*mesh_event);
}

void MCQExecutor::execute(const target::metal::MemrefCopyCommand *command) {
  auto srcIt = hostBuffers.find(command->src()->global_id());
  LOG_ASSERT(srcIt != hostBuffers.end());
  auto dstIt = hostBuffers.find(command->dst()->global_id());
  LOG_ASSERT(dstIt != hostBuffers.end());
  ttmetal::memcpy(dstIt->second, srcIt->second);
}

void MCQExecutor::execute(const target::metal::CpuCommand *command) {
  std::vector<std::vector<int64_t>> allSizesAndStrides;
  auto dataFuncPtr =
      std::function<void *(const tt::target::metal::BufferRef *)>(
          [this](const tt::target::metal::BufferRef *ref) -> void * {
            auto it = hostBuffers.find(ref->global_id());
            LOG_ASSERT(
                it != hostBuffers.end(),
                "Cannot invoke cpu op on tensor which is not in cpu tensors.");
            const Tensor &tens = it->second;
            return tens.data.get();
          });

  auto packedInputs = tt::runtime::common::packTensors(
      command->ins(), command->out(), dataFuncPtr, allSizesAndStrides);

  common::WrappedFunc func =
      dylibManager.getFunc(command->dylib_id(), command->func_name()->c_str());
  func(packedInputs.data());

  auto lastInputIt = hostBuffers.find(
      command->ins()->Get(command->ins()->size() - 1)->global_id());
  LOG_ASSERT(lastInputIt != hostBuffers.end());
  hostBuffers.insert({command->out()->global_id(), lastInputIt->second});
}

void MCQExecutor::execute(const target::metal::FinishCommand *) {
  ZoneScopedN("FinishCommand");
  distributed::Finish(*mcq);
}

std::vector<Tensor>
executeMeshDeviceProgram(distributed::MeshDevice *meshDevice,
                         const target::metal::DeviceProgram *program,
                         const std::vector<Tensor> &inputs,
                         common::DylibManager &&dylibs) {
  LOG_ASSERT(program->command_queues()->size() == 1, "Only one MCQ supported");

  MCQExecutor executor(meshDevice, program->inputs(), inputs, std::move(dylibs),
                       debug::Env::get().blockingCQ);
  for (const target::metal::CommandQueue *cq : *program->command_queues()) {
    FrameMark;
    ZoneScoped;
    std::string zoneName =
        "executeCommandQueue_mcq_" + std::to_string(cq->queue_id());
    ZoneName(zoneName.c_str(), zoneName.size());

    executor.execute(cq);

    FrameMark;
  }

  return executor.getOutputs();
}
} // namespace tt::runtime::ttmetal
