// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Microsoft Corp.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "msallreduce_cuda_operations.h"
#include <boost/asio/post.hpp>

namespace horovod {
namespace common {
std::string CublasContext::GetCublasErrorString (cublasStatus_t cublas_result) {
  switch (cublas_result) {
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return std::string("CUBLAS_STATUS_NOT_INITIALIZED");
        break;
      case CUBLAS_STATUS_ALLOC_FAILED:
        return std::string("CUBLAS_STATUS_ALLOC_FAILED");
        break;
      case CUBLAS_STATUS_INVALID_VALUE:
        return std::string("CUBLAS_STATUS_INVALID_VALUE");
        break;
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return std::string("CUBLAS_STATUS_ARCH_MISMATCH");
        break;
      case CUBLAS_STATUS_MAPPING_ERROR:
        return std::string("CUBLAS_STATUS_MAPPING_ERROR");
        break;
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return std::string("CUBLAS_STATUS_EXECUTION_FAILED");
        break;
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return std::string("CUBLAS_STATUS_INTERNAL_ERROR");
        break;
      case CUBLAS_STATUS_NOT_SUPPORTED:
        return std::string("CUBLAS_STATUS_NOT_SUPPORTED");
        break;
      case CUBLAS_STATUS_LICENSE_ERROR:
        return std::string("CUBLAS_STATUS_LICENSE_ERROR");
        break;
      default:
        return std::string("Unknown CUBLAS error!");
  }
}
void CublasContext::ErrorCheck(std::string op_name, cublasStatus_t cublas_result) {
    if (cublas_result != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error(std::string(op_name) + " failed: " + GetCublasErrorString(cublas_result));
    }
}

MsCudaAllreduceOp::MsCudaAllreduceOp(MPIContext* mpi_context, CUDAContext* cuda_context, HorovodGlobalState* global_state)
    : MsAllreduceOp(mpi_context, global_state), cuda_context_(cuda_context) {
    }

Status MsCudaAllreduceOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
      if(entries.size() < 1) {
      return Status::OK();
  }
  //TODO how do we report statuses?
  std::map<int, Status> return_statuses;
  int layerid = 0;
  int num_reductions = entries.size();
  LOG(INFO, global_state_->rank)<<"Ready to process "<<num_reductions<<" tensors in gpu";
  for (auto& e : entries) {
    boost::asio::post(*global_state_->background_thread_pool,
    [&return_statuses, this, &e, response, layerid, &entries]
    {
      LOG(INFO, global_state_->rank)<<"Begin processing gpu tensor in layer "<<layerid;
      switch (entry.output->dtype()) {
          case HOROVOD_INT8:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         DotProductImpl<int8_t>);  
          break;     
          case HOROVOD_UINT8:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         DotProductImpl<uint8_t>);  
          break;
          case HOROVOD_FLOAT16:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         DotProductImpl<MsAllreduceOp::float16>);  
          break;
          case HOROVOD_UINT16:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         DotProductImpl<uint16_t>);  
          break;
          case HOROVOD_INT16:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         DotProductImpl<int16_t>);  
          break;
          case HOROVOD_INT32:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         DotProductImpl<int32_t>);  
          break;
          case HOROVOD_INT64:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         DotProductImpl<int64_t>);  
          break;
          case HOROVOD_FLOAT32:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         DotProductImpl<float>);  
          break;
          case HOROVOD_FLOAT64:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         DotProductImpl<double>);  
          
          break;
          default:
              throw std::logic_error("MsAllreduceOp::Execute: Unsupported data type.");
      }
      LOG(INFO, global_state_->rank)<<"Done processing gpu tensor in layer "<<layerid;
      global_state_->finished_parallel_reductions++;
    });
    layerid++;
  }
  while (global_state_->finished_parallel_reductions < num_reductions) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(50));
  }
  global_state_->finished_parallel_reductions = 0;

  return Status::OK();

}

void MsCudaAllreduceOp::memcpyUtil(TensorTableEntry entry, void* dest, void* src, size_t buffer_len) {
    assert(dest != nullptr);
    assert(src != nullptr);
    LOG(INFO, global_state_->rank)<<"memcpyUtil GPU.";
    auto cuda_result = cudaMemcpyAsync(dest, src,
                                    buffer_len, 
                                    cudaMemcpyDeviceToDevice,
                                    cuda_context_->streams[global_state_->current_nccl_stream][entry.device]);
    cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
    LOG(INFO, global_state_->rank)<<"memcpyUtil GPU OK.";
}

template<typename T>
void MsCudaAllreduceOp::DotProductImpl(const T* __restrict__  a, const T* __restrict__ b, int n, double& dotProduct, double& anormsq, double& bnormsq) {
      thread_local static CublasThreadState CublasState;
}

bool MsCudaAllreduceOp::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}
}
}