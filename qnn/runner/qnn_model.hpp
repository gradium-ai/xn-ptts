// A thin wrapper over the QNN runtime: load a backend, open a context binary
// (HTP) or DLCs (CPU, GPU), and run graphs on caller-owned buffers.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

namespace phonon {

struct TensorInfo {
  std::string name;
  Qnn_Tensor_t tensor;  // as the graph declares it; name and dimensions point into this struct's fields
  std::vector<uint32_t> dims;
  size_t count = 1;  // elements
  size_t bytes = 0;  // as the graph declares it
};

struct Graph {
  std::string name;
  Qnn_GraphHandle_t handle = nullptr;
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;

  const TensorInfo& input(const std::string& name) const;
  const TensorInfo& output(const std::string& name) const;
};

class QnnModel {
 public:
  // backend_lib: libQnnHtp.so, libQnnCpu.so or libQnnGpu.so. For HTP, model_files
  // holds the context binary, whose graphs keep their own names; otherwise the
  // DLCs, composed into one context, each holding one graph that is known by
  // the name paired with its file.
  QnnModel(const std::string& backend_lib, const std::string& system_lib,
           const std::vector<std::pair<std::string, std::string>>& model_files);
  ~QnnModel();
  QnnModel(const QnnModel&) = delete;
  QnnModel& operator=(const QnnModel&) = delete;

  const Graph& graph(const std::string& name) const;

  // A buffer for `tensor`, owned by the model: fp16 for a float tensor, else the
  // tensor's own type. On the HTP it is shared memory registered with QNN, so
  // execute() hands it to the NPU without copying it; elsewhere it is ordinary
  // memory. It may also be bound to any other tensor of the same shape and type.
  void* alloc(const TensorInfo& tensor);

  // Binds buffers by tensor name; every input and output must be given. Float
  // buffers are fp16; a tensor the graph declares fp32 is converted on the way
  // in and out. Other buffers are exactly the graph's size. Buffers from alloc()
  // are passed to the backend as they are.
  void execute(const Graph& graph, const std::map<std::string, const void*>& inputs,
               const std::map<std::string, void*>& outputs);

  // HTP only: hold the NPU at its highest clock and poll for results, rather than
  // letting it ramp up and sleep between calls. A no-op on other backends.
  void set_performance_mode(bool burst);

  // With PHONON_QNN_PROFILE=1 in the environment, the HTP's own timing of each
  // call, summed per graph: time on the NPU, and the RPC time around it.
  struct Timing {
    int calls = 0;
    double npu_us = 0, host_rpc_us = 0, htp_rpc_us = 0;
  };
  const std::map<std::string, Timing>& timing() const { return timing_; }
  bool profiling() const { return profile_ != nullptr; }

 private:
  void cleanup() noexcept;
  void load_context_binary(const std::string& path);
  void load_dlcs(const std::vector<std::pair<std::string, std::string>>& files);
  void add_graph(const std::string& name, const char* qnn_name, uint32_t n_in, const Qnn_Tensor_t* in,
                 uint32_t n_out, const Qnn_Tensor_t* out);

  void* backend_lib_ = nullptr;
  void* system_lib_ = nullptr;
  const QnnInterface_t* provider_ = nullptr;
  QNN_INTERFACE_VER_TYPE qnn_{};
  QNN_SYSTEM_INTERFACE_VER_TYPE sys_{};
  bool is_htp_ = false;
  Qnn_LogHandle_t log_ = nullptr;
  Qnn_BackendHandle_t backend_ = nullptr;
  Qnn_DeviceHandle_t device_ = nullptr;
  Qnn_ContextHandle_t context_ = nullptr;
  std::map<std::string, Graph> graphs_;
  uint32_t power_config_id_ = 0;
  bool has_power_config_ = false;

  // Shared memory (rpcmem from libcdsprpc), HTP only.
  struct Shared {
    Qnn_MemHandle_t handle = nullptr;
    size_t bytes = 0;
  };
  void* cdsprpc_ = nullptr;
  void* (*rpcmem_alloc_)(int, uint32_t, int) = nullptr;
  void (*rpcmem_free_)(void*) = nullptr;
  int (*rpcmem_to_fd_)(void*) = nullptr;
  std::map<void*, Shared> shared_;
  std::vector<void*> plain_;  // alloc()'s ordinary buffers

  Qnn_ProfileHandle_t profile_ = nullptr;
  std::map<std::string, Timing> timing_;
  void collect_profile(const std::string& graph);
};

}  // namespace phonon
