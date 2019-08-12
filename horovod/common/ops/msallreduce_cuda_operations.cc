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
  for (auto& entry : entries) {
    boost::asio::post(*global_state_->background_thread_pool,
    [&return_statuses, this, &entry, response, layerid, &entries]
    {
      void* buffer_data;
      int buffer_len;
      void* recv_buffer;

      buffer_data = (void*) entry.tensor->data();

      buffer_len = entry.output->size();

      if(entry.tensor->data() == entry.output->data()) {
          // Get the temp buffer to be used for the Op
          global_state_->buffer_lock.lock();
          assert(!global_state_->temp_buffers.empty());
          buffer_manager = global_state_->temp_buffers.front();
          global_state_->temp_buffers.pop();
          global_state_->buffer_lock.unlock();

          // TODO: Maybe add before and after callbacks to timeline?
          Status status = buffer_manager.InitializeBuffer(
              buffer_len,
              entry.device, entry.context,
              global_state_->current_nccl_stream,
              [](){},
              [](){},
              [](int64_t& size, int64_t& threshold){return size >= threshold;});

          if (!status.ok()) {
              throw std::logic_error("MsAllreduceOp::Execute_helper: Initialize buffer failed.");
              return;
          }

          auto& buffer = buffer_manager.GetBuffer(entry.device, entry.context->framework(), global_state_->current_nccl_stream);
          recv_buffer = const_cast<void*>(buffer->AccessData(entry.context));
      }
      else {
          recv_buffer = (void*) entry.output->data();
      }
      LOG(INFO, global_state_->rank)<<"Begin to process gpu tensor with size "<<entry.tensor->size()<<" into output buffer with size "<<entry.output->size();
      
      MPI_Comm* node_comm = NULL;
      if (global_state_->rank_log_size != 0) {
          node_comm = &global_state_->reduction_comms[global_state_->rank_log_size-1];
      }

      LOG(INFO, global_state_->rank)<<"Begin processing gpu tensor in layer "<<layerid;
      switch (entry.output->dtype()) {
          case HOROVOD_INT8:
          //TODO new parasail
            MsAllreduce_Internal((int8_t*) buffer_data,
                            (int8_t*) recv_buffer,
                            buffer_len,
                            node_comm,
                            layerid,
                            entry,
                            DotProductImpl<int8_t>,
                            ScaleAddImpl<int8_t>);  
          break;     
          case HOROVOD_UINT8:
          //TODO new parasail
            MsAllreduce_Internal((uint8_t*) buffer_data,
                            (uint8_t*) recv_buffer,
                            buffer_len,
                            node_comm,
                            layerid,
                            entry,
                            DotProductImpl<uint8_t>,
                            ScaleAddImpl<uint8_t>);  
          break;
          case HOROVOD_FLOAT16:
          //TODO new parasail
            MsAllreduce_Internal((MsAllreduceOp::float16*) buffer_data,
                            (MsAllreduceOp::float16*) recv_buffer,
                            buffer_len,
                            node_comm,
                            layerid,
                            entry,
                            DotProductImpl<MsAllreduceOp::float16>,
                            ScaleAddImpl<MsAllreduceOp::float16>);  
          break;
          case HOROVOD_UINT16:
          //TODO new parasail
            MsAllreduce_Internal((uint16_t*) buffer_data,
                            (uint16_t*) recv_buffer,
                            buffer_len,
                            node_comm,
                            layerid,
                            entry,
                            DotProductImpl<uint16_t>,
                            ScaleAddImpl<uint16_t>);  
          break;
          case HOROVOD_INT16:
          //TODO new parasail
            MsAllreduce_Internal((int16_t*) buffer_data,
                            (int16_t*) recv_buffer,
                            buffer_len,
                            node_comm,
                            layerid,
                            entry,
                            DotProductImpl<int16_t>,
                            ScaleAddImpl<int16_t>);  
          break;
          case HOROVOD_INT32:
          //TODO new parasail
            MsAllreduce_Internal((int32_t*) buffer_data,
                            (int32_t*) recv_buffer,
                            buffer_len,
                            node_comm,
                            layerid,
                            entry,
                            DotProductImpl<int32_t>,
                            ScaleAddImpl<int32_t>);  
          break;
          case HOROVOD_INT64:
          //TODO new parasail
            MsAllreduce_Internal((int64_t*) buffer_data,
                            (int64_t*) recv_buffer,
                            buffer_len,
                            node_comm,
                            layerid,
                            entry,
                            DotProductImpl<int64_t>,
                            ScaleAddImpl<int64_t>);  
          break;
          case HOROVOD_FLOAT32:
          //TODO new parasail
            MsAllreduce_Internal((float*) buffer_data,
                            (float*) recv_buffer,
                            buffer_len,
                            node_comm,
                            layerid,
                            entry,
                            DotProductImpl<float>,
                            ScaleAddImpl<float>);  
          break;
          case HOROVOD_FLOAT64:
          //TODO new parasail
            MsAllreduce_Internal((double*) buffer_data,
                            (double*) recv_buffer,
                            buffer_len,
                            node_comm,
                            layerid,
                            entry,
                            DotProductImpl<double>,
                            ScaleAddImpl<double>);  
          
          break;
          default:
              throw std::logic_error("MsAllreduceOp::Execute: Unsupported data type.");
      }
      LOG(INFO, global_state_->rank)<<"Done processing tensor in layer "<<layerid;
      if(entry.tensor->data() == entry.output->data()) {
        // Return the buffer back into the pool of available buffers
        global_state_->buffer_lock.lock();
        global_state_->temp_buffers.push(buffer_manager);
        global_state_->buffer_lock.unlock();
      }

      memcpyUtil(entry, (void *) entry.output->data(), (void *) entry.tensor->data(), (size_t) entry.tensor->size());
      LOG(INFO, global_state_->rank)<<"Finished ms gpu allreduction, exiting operation";

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

//TODO CRITICAL! fix the force casting to double, this results in divide-by-zero exception
template<typename T>
void MsCudaAllreduceOp::DotProductImpl(const T* __restrict__  a, const T* __restrict__ b, int n, float& dotProduct, float& anormsq, float& bnormsq, HorovodGlobalState *global_state) {
  cublasHandle_t handle = getCublasThreadState().cublasHandle;
  
  auto adotbstatus = cublasDotEx(handle, n, (float *)a, CUDA_R_32F, 1, (float *)b, CUDA_R_32F, 1, &dotProduct, CUDA_R_32F, CUDA_R_32F);
  CublasContext::ErrorCheck("a cublasdot b", adotbstatus);
  
  auto adotastatus = cublasDotEx(handle, n, (float *)a, CUDA_R_32F, 1, (float *)a, CUDA_R_32F, 1, &dotProduct, CUDA_R_32F, CUDA_R_32F);
  CublasContext::ErrorCheck("a cublasdot a", adotastatus);

  auto bdotbstatus = cublasDotEx(handle, n, (float *)b, CUDA_R_32F, 1, (float *)b, CUDA_R_32F, 1, &dotProduct, CUDA_R_32F, CUDA_R_32F);
  CublasContext::ErrorCheck("b cublasdot b", bdotbstatus);
}

//TODO CRITICAL! fix the force casting to double, this results in divide-by-zero exception
template<typename T>
void MsCudaAllreduceOp::ScaleAddImpl(int n, float acoeff, T* __restrict__ a, float bcoeff, T* __restrict__ b, HorovodGlobalState *global_state) {
  cublasHandle_t handle = getCublasThreadState().cublasHandle;
  
  auto scaleStatus = cublasScalEx(handle, n, &acoeff, CUDA_R_32F, (float *)a, CUDA_R_32F, 1, CUDA_R_32F);
  CublasContext::ErrorCheck("cublasDscal", scaleStatus);
  
  auto axpyStatus = cublasAxpyEx(handle, n, &bcoeff, CUDA_R_32F, (float *)b, CUDA_R_32F, 1, (float *)a, CUDA_R_32F, 1, CUDA_R_32F);
  CublasContext::ErrorCheck("cublasDaxpy", axpyStatus);
}

CublasThreadState MsCudaAllreduceOp::getCublasThreadState() {
  thread_local static CublasThreadState CublasState;
  return CublasState;
}

bool MsCudaAllreduceOp::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}
}
}