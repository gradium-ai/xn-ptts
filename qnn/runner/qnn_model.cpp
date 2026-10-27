#include "qnn_model.hpp"

#include <dlfcn.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "HTP/QnnHtpCommon.h"
#include "f16.hpp"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "HTP/QnnHtpProfile.h"
#include "QnnMem.h"
#include "QnnProfile.h"

namespace phonon {
namespace {

void check(Qnn_ErrorHandle_t status, const char* what) {
  if (status != QNN_SUCCESS) throw std::runtime_error(std::string(what) + " failed: QNN error " + std::to_string(status));
}

// PHONON_QNN_LOG=verbose (or debug, info) shows more of QNN's own logging.
QnnLog_Level_t log_level() {
  const char* v = std::getenv("PHONON_QNN_LOG");
  if (!v) return QNN_LOG_LEVEL_WARN;
  std::string s = v;
  if (s == "verbose") return QNN_LOG_LEVEL_VERBOSE;
  if (s == "debug") return QNN_LOG_LEVEL_DEBUG;
  if (s == "info") return QNN_LOG_LEVEL_INFO;
  return QNN_LOG_LEVEL_WARN;
}

void log_callback(const char* fmt, QnnLog_Level_t level, uint64_t, va_list args) {
  if (level > log_level()) return;
  std::fprintf(stderr, "[qnn] ");
  std::vfprintf(stderr, fmt, args);
  std::fprintf(stderr, "\n");
}

void* open_lib(const std::string& path) {
  void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!h) throw std::runtime_error("dlopen " + path + ": " + dlerror());
  return h;
}

size_t dtype_size(Qnn_DataType_t t) {
  switch (t) {
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
      return 4;
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
      return 2;
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_BOOL_8:
      return 1;
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
      return 8;
    default:
      throw std::runtime_error("unsupported tensor data type " + std::to_string(t));
  }
}

// v1 and v2 tensors share their leading fields; both are read through v1.
const Qnn_TensorV1_t& v1(const Qnn_Tensor_t& t) { return t.v1; }
Qnn_TensorV1_t& v1(Qnn_Tensor_t& t) { return t.v1; }

std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<char> buf(f.tellg());
  f.seekg(0);
  f.read(buf.data(), buf.size());
  return buf;
}

}  // namespace

const TensorInfo& Graph::input(const std::string& n) const {
  for (const auto& t : inputs)
    if (t.name == n) return t;
  throw std::runtime_error("graph " + name + " has no input " + n);
}

const TensorInfo& Graph::output(const std::string& n) const {
  for (const auto& t : outputs)
    if (t.name == n) return t;
  throw std::runtime_error("graph " + name + " has no output " + n);
}

QnnModel::QnnModel(const std::string& backend_lib, const std::string& system_lib,
                   const std::vector<std::pair<std::string, std::string>>& model_files) {
  try {
    backend_lib_ = open_lib(backend_lib);
    system_lib_ = open_lib(system_lib);

    using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
    auto get_providers = reinterpret_cast<GetProviders>(dlsym(backend_lib_, "QnnInterface_getProviders"));
    if (!get_providers) throw std::runtime_error("no QnnInterface_getProviders in " + backend_lib);
    const QnnInterface_t** providers = nullptr;
    uint32_t n = 0;
    check(get_providers(&providers, &n), "QnnInterface_getProviders");
    bool found = false;
    for (uint32_t i = 0; i < n; i++) {
      if (providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
          providers[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
        provider_ = providers[i];
        qnn_ = providers[i]->QNN_INTERFACE_VER_NAME;
        is_htp_ = providers[i]->backendId == QNN_BACKEND_ID_HTP;
        found = true;
        break;
      }
    }
    if (!found) throw std::runtime_error("no compatible QNN interface in " + backend_lib);

    using GetSysProviders = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);
    auto get_sys = reinterpret_cast<GetSysProviders>(dlsym(system_lib_, "QnnSystemInterface_getProviders"));
    if (!get_sys) throw std::runtime_error("no QnnSystemInterface_getProviders in " + system_lib);
    const QnnSystemInterface_t** sys_providers = nullptr;
    check(get_sys(&sys_providers, &n), "QnnSystemInterface_getProviders");
    found = false;
    for (uint32_t i = 0; i < n; i++) {
      if (sys_providers[i]->systemApiVersion.major == QNN_SYSTEM_API_VERSION_MAJOR &&
          sys_providers[i]->systemApiVersion.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
        sys_ = sys_providers[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
        found = true;
        break;
      }
    }
    if (!found) throw std::runtime_error("no compatible QNN system interface in " + system_lib);

    check(qnn_.logCreate(log_callback, log_level(), &log_), "logCreate");
    check(qnn_.backendCreate(log_, nullptr, &backend_), "backendCreate");
    if (qnn_.deviceCreate) {
      Qnn_ErrorHandle_t s = qnn_.deviceCreate(log_, nullptr, &device_);
      if (s != QNN_SUCCESS && s != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) check(s, "deviceCreate");
    }
    if (is_htp_) {
      if (model_files.size() != 1) throw std::runtime_error("HTP takes one context binary");
      load_context_binary(model_files[0].second);
      // Shared memory for zero-copy buffers; without it alloc() falls back to ordinary memory.
      cdsprpc_ = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
      if (cdsprpc_) {
        rpcmem_alloc_ = reinterpret_cast<decltype(rpcmem_alloc_)>(dlsym(cdsprpc_, "rpcmem_alloc"));
        rpcmem_free_ = reinterpret_cast<decltype(rpcmem_free_)>(dlsym(cdsprpc_, "rpcmem_free"));
        rpcmem_to_fd_ = reinterpret_cast<decltype(rpcmem_to_fd_)>(dlsym(cdsprpc_, "rpcmem_to_fd"));
        if (!rpcmem_alloc_ || !rpcmem_free_ || !rpcmem_to_fd_) rpcmem_alloc_ = nullptr;
      }
    } else {
      load_dlcs(model_files);
    }
    const char* profile = std::getenv("PHONON_QNN_PROFILE");
    if (profile && std::string(profile) == "1" && qnn_.profileCreate)
      check(qnn_.profileCreate(backend_, QNN_PROFILE_LEVEL_BASIC, &profile_), "profileCreate");
  } catch (...) { cleanup(); throw; }
}

QnnModel::~QnnModel() { cleanup(); }

void QnnModel::cleanup() noexcept {
  set_performance_mode(false);
  for (auto& [ptr, shared] : shared_) {
    qnn_.memDeRegister(&shared.handle, 1);
    rpcmem_free_(ptr);
  }
  for (void* ptr : plain_) std::free(ptr);
  if (profile_) qnn_.profileFree(profile_);
  if (context_) qnn_.contextFree(context_, nullptr);
  if (device_ && qnn_.deviceFree) qnn_.deviceFree(device_);
  if (backend_) qnn_.backendFree(backend_);
  if (log_) qnn_.logFree(log_);
  // The libraries stay loaded: unloading the HTP backend at exit can hang on some devices.
}

void QnnModel::add_graph(const std::string& name, const char* qnn_name, uint32_t n_in, const Qnn_Tensor_t* in,
                         uint32_t n_out, const Qnn_Tensor_t* out) {
  // Built in place: map nodes never move, so the tensors can point at their own
  // name and dims once every copy is made.
  Graph& g = graphs_[name];
  g.name = name;
  check(qnn_.graphRetrieve(context_, qnn_name, &g.handle), "graphRetrieve");
  auto copy = [](const Qnn_Tensor_t& t) {
    TensorInfo info;
    info.name = v1(t).name;
    info.tensor = t;
    info.dims.assign(v1(t).dimensions, v1(t).dimensions + v1(t).rank);
    for (uint32_t d : info.dims) info.count *= d;
    info.bytes = info.count * dtype_size(v1(t).dataType);
    return info;
  };
  for (uint32_t i = 0; i < n_in; i++) g.inputs.push_back(copy(in[i]));
  for (uint32_t i = 0; i < n_out; i++) g.outputs.push_back(copy(out[i]));
  for (auto* list : {&g.inputs, &g.outputs}) {
    for (auto& t : *list) {
      v1(t.tensor).name = t.name.c_str();
      v1(t.tensor).dimensions = t.dims.data();
      if (log_level() >= QNN_LOG_LEVEL_INFO) {
        std::fprintf(stderr, "[phonon] %s %s %s: dtype 0x%x, %zu bytes, dims", g.name.c_str(),
                     list == &g.inputs ? "input" : "output", t.name.c_str(), unsigned(v1(t.tensor).dataType), t.bytes);
        for (uint32_t d : t.dims) std::fprintf(stderr, " %u", d);
        std::fprintf(stderr, "\n");
      }
    }
  }
}

void QnnModel::load_context_binary(const std::string& path) {
  std::vector<char> blob = read_file(path);
  QnnSystemContext_Handle_t sys_ctx = nullptr;
  check(sys_.systemContextCreate(&sys_ctx), "systemContextCreate");
  auto release = [&](void* ctx) { sys_.systemContextFree(ctx); };
  std::unique_ptr<void, decltype(release)> guard(sys_ctx, release);
  const QnnSystemContext_BinaryInfo_t* info = nullptr;
  Qnn_ContextBinarySize_t info_size = 0;
  check(sys_.systemContextGetBinaryInfo(sys_ctx, blob.data(), blob.size(), &info, &info_size),
        "systemContextGetBinaryInfo");
  check(qnn_.contextCreateFromBinary(backend_, device_, nullptr, blob.data(), blob.size(), &context_, nullptr),
        "contextCreateFromBinary");

  uint32_t n_graphs = 0;
  const QnnSystemContext_GraphInfo_t* graphs = nullptr;
  switch (info->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
      n_graphs = info->contextBinaryInfoV1.numGraphs;
      graphs = info->contextBinaryInfoV1.graphs;
      break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
      n_graphs = info->contextBinaryInfoV2.numGraphs;
      graphs = info->contextBinaryInfoV2.graphs;
      break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
      n_graphs = info->contextBinaryInfoV3.numGraphs;
      graphs = info->contextBinaryInfoV3.graphs;
      break;
    default:
      throw std::runtime_error("unknown context binary info version");
  }
  // Every graph info version starts with the same name, inputs and outputs fields.
  for (uint32_t i = 0; i < n_graphs; i++) {
    const auto& g = graphs[i].graphInfoV1;
    add_graph(g.graphName, g.graphName, g.numGraphInputs, g.graphInputs, g.numGraphOutputs, g.graphOutputs);
  }
}

void QnnModel::load_dlcs(const std::vector<std::pair<std::string, std::string>>& files) {
  check(qnn_.contextCreate(backend_, device_, nullptr, &context_), "contextCreate");
  Qnn_LogHandle_t sys_log = nullptr;
  check(sys_.systemLogCreate(log_callback, log_level(), &sys_log), "systemLogCreate");
  for (const auto& [name, path] : files) {
    QnnSystemDlc_Handle_t dlc = nullptr;
    check(sys_.systemDlcCreateFromFile(sys_log, path.c_str(), &dlc), ("systemDlcCreateFromFile " + path).c_str());
    QnnSystemContext_GraphInfo_t* infos = nullptr;
    uint32_t n = 0;
    check(sys_.systemDlcComposeGraphs(dlc, nullptr, 0, backend_, context_, *provider_,
                                      QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1, &infos, &n),
          ("systemDlcComposeGraphs " + path).c_str());
    if (n != 1) throw std::runtime_error(path + ": expected one graph, found " + std::to_string(n));
    for (uint32_t i = 0; i < n; i++) {
      const auto& g = infos[i].graphInfoV1;
      Qnn_GraphHandle_t handle = nullptr;
      check(qnn_.graphRetrieve(context_, g.graphName, &handle), "graphRetrieve");
      check(qnn_.graphFinalize(handle, nullptr, nullptr), "graphFinalize");
      add_graph(name, g.graphName, g.numGraphInputs, g.graphInputs, g.numGraphOutputs, g.graphOutputs);
      free(const_cast<char*>(g.graphName));
      free(g.graphInputs);
      free(g.graphOutputs);
    }
    free(infos);
    sys_.systemDlcFree(dlc);
  }
  sys_.systemLogFree(sys_log);
}

const Graph& QnnModel::graph(const std::string& name) const {
  auto it = graphs_.find(name);
  if (it == graphs_.end()) throw std::runtime_error("no graph " + name);
  return it->second;
}

void* QnnModel::alloc(const TensorInfo& info) {
  const Qnn_DataType_t type = v1(info.tensor).dataType;
  const bool is_float = type == QNN_DATATYPE_FLOAT_16 || type == QNN_DATATYPE_FLOAT_32;
  const size_t bytes = is_float ? info.count * sizeof(f16) : info.bytes;
  // Shared memory only when the host's layout is the tensor's: an fp32 tensor
  // (CPU DLCs) goes through execute()'s conversion instead.
  if (rpcmem_alloc_ && (!is_float || type == QNN_DATATYPE_FLOAT_16)) {
    void* ptr = rpcmem_alloc_(25 /* RPCMEM_HEAP_ID_SYSTEM */, 1 /* RPCMEM_DEFAULT_FLAGS */, int(bytes));
    if (ptr) {
      Qnn_MemDescriptor_t desc;
      std::memset(&desc, 0, sizeof(desc));
      desc.memShape = {v1(info.tensor).rank, v1(info.tensor).dimensions, nullptr};
      desc.dataType = type;
      desc.memType = QNN_MEM_TYPE_ION;
      desc.ionInfo.fd = rpcmem_to_fd_(ptr);
      Qnn_MemHandle_t handle = nullptr;
      if (desc.ionInfo.fd >= 0 && qnn_.memRegister(context_, &desc, 1, &handle) == QNN_SUCCESS) {
        std::memset(ptr, 0, bytes);
        shared_[ptr] = {handle, bytes};
        return ptr;
      }
      rpcmem_free_(ptr);
    }
  }
  void* ptr = std::calloc(1, bytes);
  if (!ptr) throw std::bad_alloc();
  plain_.push_back(ptr);
  return ptr;
}

void QnnModel::collect_profile(const std::string& graph) {
  const QnnProfile_EventId_t* events = nullptr;
  uint32_t n = 0;
  if (qnn_.profileGetEvents(profile_, &events, &n) != QNN_SUCCESS) return;
  Timing& t = timing_[graph];
  t.calls++;
  auto take = [&](QnnProfile_EventId_t id) {
    QnnProfile_EventData_t data;
    if (qnn_.profileGetEventData(id, &data) != QNN_SUCCESS) return;
    if (data.type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_ACCEL_TIME_MICROSEC) t.npu_us += double(data.value);
    if (data.type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_HOST_RPC_TIME_MICROSEC) t.host_rpc_us += double(data.value);
    if (data.type == QNN_HTP_PROFILE_EVENTTYPE_GRAPH_EXECUTE_HTP_RPC_TIME_MICROSEC) t.htp_rpc_us += double(data.value);
  };
  for (uint32_t i = 0; i < n; i++) {
    take(events[i]);
    const QnnProfile_EventId_t* sub = nullptr;
    uint32_t m = 0;
    if (qnn_.profileGetSubEvents(events[i], &sub, &m) == QNN_SUCCESS)
      for (uint32_t j = 0; j < m; j++) take(sub[j]);
  }
}

void QnnModel::execute(const Graph& g, const std::map<std::string, const void*>& inputs,
                       const std::map<std::string, void*>& outputs) {
  // fp32 tensors get a staging buffer, filled from or copied back to the caller's fp16.
  std::vector<std::vector<float>> staging;
  std::vector<std::pair<const std::vector<float>*, f16*>> copy_back;
  auto bind = [&](const std::vector<TensorInfo>& infos, auto& buffers, const char* kind, bool is_output) {
    std::vector<Qnn_Tensor_t> tensors;
    for (const auto& info : infos) {
      auto it = buffers.find(info.name);
      if (it == buffers.end()) throw std::runtime_error(std::string("missing ") + kind + " " + info.name);
      void* data = const_cast<void*>(static_cast<const void*>(it->second));
      auto shared = shared_.find(data);
      if (shared != shared_.end()) {
        Qnn_Tensor_t t = info.tensor;
        v1(t).memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        v1(t).memHandle = shared->second.handle;
        tensors.push_back(t);
        continue;
      }
      if (v1(info.tensor).dataType == QNN_DATATYPE_FLOAT_32) {
        auto& buf = staging.emplace_back(info.count);
        auto* host = static_cast<f16*>(data);
        if (is_output) copy_back.push_back({&buf, host});
        else for (size_t i = 0; i < buf.size(); i++) buf[i] = float(host[i]);
        data = buf.data();
      }
      Qnn_Tensor_t t = info.tensor;
      v1(t).memType = QNN_TENSORMEMTYPE_RAW;
      v1(t).clientBuf.data = data;
      v1(t).clientBuf.dataSize = static_cast<uint32_t>(info.bytes);
      tensors.push_back(t);
    }
    return tensors;
  };
  staging.reserve(g.inputs.size() + g.outputs.size());  // the bound pointers must not move
  std::vector<Qnn_Tensor_t> in = bind(g.inputs, inputs, "input", false);
  std::vector<Qnn_Tensor_t> out = bind(g.outputs, outputs, "output", true);
  check(qnn_.graphExecute(g.handle, in.data(), in.size(), out.data(), out.size(), profile_, nullptr),
        ("graphExecute " + g.name).c_str());
  if (profile_) collect_profile(g.name);
  for (auto& [buf, host] : copy_back)
    for (size_t i = 0; i < buf->size(); i++) host[i] = f16((*buf)[i]);
}

void QnnModel::set_performance_mode(bool burst) {
  if (!is_htp_ || !qnn_.deviceGetInfrastructure) return;
  QnnDevice_Infrastructure_t infra = nullptr;
  if (qnn_.deviceGetInfrastructure(&infra) != QNN_SUCCESS || !infra) return;
  auto* htp = static_cast<QnnHtpDevice_Infrastructure_t*>(infra);
  if (htp->infraType != QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF) return;
  auto& perf = htp->perfInfra;
  if (!burst) {
    if (has_power_config_) perf.destroyPowerConfigId(power_config_id_);
    has_power_config_ = false;
    return;
  }
  if (!has_power_config_) {
    if (perf.createPowerConfigId(0, 0, &power_config_id_) != QNN_SUCCESS) return;
    has_power_config_ = true;
  }
  QnnHtpPerfInfrastructure_PowerConfig_t cfg;
  std::memset(&cfg, 0, sizeof(cfg));
  cfg.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
  auto& d = cfg.dcvsV3Config;
  d.contextId = power_config_id_;
  d.setDcvsEnable = 1;
  d.dcvsEnable = 0;
  d.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  d.setSleepLatency = 1;
  d.sleepLatency = 40;  // microseconds
  d.setSleepDisable = 1;
  d.sleepDisable = 1;
  d.setBusParams = 1;
  d.busVoltageCornerMin = d.busVoltageCornerTarget = d.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  d.setCoreParams = 1;
  d.coreVoltageCornerMin = d.coreVoltageCornerTarget = d.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
  // Poll for results instead of sleeping, and keep the RPC wake-up latency low:
  // the loop makes one short call per 80 ms frame.
  QnnHtpPerfInfrastructure_PowerConfig_t latency, polling;
  std::memset(&latency, 0, sizeof(latency));
  std::memset(&polling, 0, sizeof(polling));
  latency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
  latency.rpcControlLatencyConfig = 100;  // microseconds
  polling.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
  polling.rpcPollingTimeConfig = 9999;  // microseconds, the maximum
  const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[] = {&cfg, &latency, &polling, nullptr};
  perf.setPowerConfig(power_config_id_, cfgs);
}

}  // namespace phonon
