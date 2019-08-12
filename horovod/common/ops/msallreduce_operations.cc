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

#include "msallreduce_operations.h"
#include <boost/asio/post.hpp>

namespace horovod {
namespace common {

MsAllreduceOp::MsAllreduceOp(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : PointToPointOp(mpi_context, global_state) {}

Status MsAllreduceOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  if(entries.size() < 1) {
      return Status::OK();
  }
  //TODO how do we report statuses?
  std::map<int, Status> return_statuses;
  int layerid = 0;
  int num_reductions = entries.size();
  LOG(INFO, global_state_->rank)<<"Ready to process "<<num_reductions<<" tensors";
  for (auto& e : entries) {
    boost::asio::post(*global_state_->background_thread_pool,
    [&return_statuses, this, &e, response, layerid, &entries]
    {
      LOG(INFO, global_state_->rank)<<"Begin processing tensor in layer "<<layerid;
      switch (entry.output->dtype()) {
          case HOROVOD_INT8:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         ComputeDotAndNormSqrds<int8_t>);  
          break;     
          case HOROVOD_UINT8:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         ComputeDotAndNormSqrds<uint8_t>);  
          break;
          case HOROVOD_FLOAT16:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         ComputeDotAndNormSqrds<MsAllreduceOp::float16>);  
          break;
          case HOROVOD_UINT16:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         ComputeDotAndNormSqrds<uint16_t>);  
          break;
          case HOROVOD_INT16:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         ComputeDotAndNormSqrds<int16_t>);  
          break;
          case HOROVOD_INT32:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         ComputeDotAndNormSqrds<int32_t>);  
          break;
          case HOROVOD_INT64:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         ComputeDotAndNormSqrds<int64_t>);  
          break;
          case HOROVOD_FLOAT32:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         ComputeDotAndNormSqrds<float>);  
          break;
          case HOROVOD_FLOAT64:
          //TODO new parasail
          Execute_helper(return_statuses,
                         e,
                         response,
                         layerid,
                         ComputeDotAndNormSqrds<double>);  
          
          break;
          default:
              throw std::logic_error("MsAllreduceOp::Execute: Unsupported data type.");
      }
      LOG(INFO, global_state_->rank)<<"Done processing tensor in layer "<<layerid;
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

void MsAllreduceOp::memcpyUtil(TensorTableEntry entry, void* dest, void* src, size_t buffer_len) {
    assert(dest != nullptr);
    assert(src != nullptr);
    LOG(INFO, global_state_->rank)<<"memcpyUtil CPU.";
    memcpy(dest, src, buffer_len);
}

template<typename T, typename F>
void MsAllreduceOp::Execute_helper(std::map<int, Status>& return_status, 
                                   TensorTableEntry& entry, 
                                   const Response& response, 
                                   int layerid,
                                   F dotProdFunc){
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
  LOG(INFO, global_state_->rank)<<"Begin to process tensor with size "<<entry.tensor->size()<<" into output buffer with size "<<entry.output->size();
  
  MPI_Comm* node_comm = NULL;
  if (global_state_->rank_log_size != 0) {
	node_comm = &global_state_->reduction_comms[global_state_->rank_log_size-1];
  }

    MsAllreduce_Internal((T*) buffer_data,
                    (T*) recv_buffer,
                    buffer_len,
                    node_comm,
                    layerid,
                    entry,
                    dotProdFunc<T>);  

  if(entry.tensor->data() == entry.output->data()) {
    // Return the buffer back into the pool of available buffers
    global_state_->buffer_lock.lock();
    global_state_->temp_buffers.push(buffer_manager);
    global_state_->buffer_lock.unlock();
  }

  memcpyUtil(entry, (void *) entry.output->data(), (void *) entry.tensor->data(), (size_t) entry.tensor->size());
  LOG(INFO, global_state_->rank)<<"Finished ms allreduction, exiting operation";
}

bool MsAllreduceOp::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

// TODO new parasail algo begin
template<typename T, typename F>
void MsAllreduceOp::MsAllreduce_Internal(T* grad_buffer, T* recv_buffer, int buffer_length, MPI_Comm* node_comm, int message_tag, TensorTableEntry entry, F dotProdFunc) {
  int count = buffer_length / sizeof(T);
  int local_rank = 0;
  MPI_Comm_rank(global_state_->local_comm, &local_rank);
  SyncLocalReduce(grad_buffer, recv_buffer, count, global_state_->local_comm, message_tag, entry, dotProdFunc<T>);
  if (local_rank == 0 && node_comm != NULL) {
    SyncAllreduce(grad_buffer, recv_buffer, count, *node_comm, global_state_->reduction_comms, message_tag, dotProdFunc<T>);
  }
  SyncLocalBroadcast(grad_buffer, recv_buffer, count, global_state_->local_comm, message_tag);
  
}

template<typename T>
void static MsAllreduceOp::ComputeDotAndNormSqrds(const T* __restrict__  a, const T* __restrict__ b, int n, double& dotProduct, double& anormsq, double& bnormsq, HorovodGlobalState *global_state) {
    dotProduct = 0.;
    anormsq = 0.;
    bnormsq = 0.;
    LOG(INFO, global_state->rank)<<"Entering ComputeDotAndNormSqrds";

    for (int i = 0; i < n; i++) {
        dotProduct += a[i] * b[i];
        anormsq += a[i] * a[i];
        bnormsq += b[i] * b[i];
    }
    LOG(INFO, global_state->rank)<<"Returning ComputeDotAndNormSqrds";
}

template<typename T>
void MsAllreduceOp::ScaledAdd(int n, double acoeff, T* __restrict__ a, double bcoeff, T* __restrict__ b) {
    for (int i = 0; i < n; i++) {
        a[i] = acoeff * a[i] + bcoeff * b[i];
    }
}

template<typename T, typename F>
void MsAllreduceOp::PairwiseReduceWithComm(T* a, T* b, int count, int message_tag, MPI_Comm& comm, bool isLeftNeighbor, F dotProdFunc) {
    double dotProduct = 0.f;
    double anormsq = 0.f;
    double bnormsq = 0.f;
    LOG(INFO, global_state_->rank)<<"Computing dot product.";
    dotProdFunc(a, b, count, dotProduct, anormsq, bnormsq, global_state_);
    LOG(INFO, global_state_->rank)<<"Computed dot product.";
    double reduce_vals[3];
    if (isLeftNeighbor) { 
        reduce_vals[0] = anormsq;
        reduce_vals[1] = bnormsq;
    } else {
        reduce_vals[1] = anormsq;
        reduce_vals[0] = bnormsq;
    }
    reduce_vals[2] = dotProduct;
    // TODO replace this with something else
    MPI_Allreduce(MPI_IN_PLACE, reduce_vals, 3, MPI_DOUBLE, MPI_SUM, comm);
    LOG(INFO, global_state_->rank)<<"Performed mpi allreduce.";

    if (isLeftNeighbor) { 
        anormsq = reduce_vals[0];
        bnormsq = reduce_vals[1];
    } else {
        anormsq = reduce_vals[1];
        bnormsq = reduce_vals[0];
    }
    dotProduct = reduce_vals[2];

    double acoeff = 1;
    double bcoeff = 1;
    if (anormsq != 0)
        acoeff = 1.0 - dotProduct / anormsq * 0.5;
    if (bnormsq != 0)
        bcoeff = 1.0 - dotProduct / bnormsq * 0.5;

    // a = acoeff * a + bcoeff * b
    ScaledAdd(count, acoeff, a, bcoeff, b);
    LOG(INFO, global_state_->rank)<<"Performed ScaledAdd.";
}

template <typename T>
void MsAllreduceOp::SyncLocalBroadcast(T *grad_buffer, T *recv_buffer, int count, MPI_Comm communicator, int message_tag)
{
    int rank;
    int size;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &size);
    MPI_Request* reqs = new MPI_Request[(size-1)*2];
    int num_reqs = 0;
    if (rank == 0){
        for (int i = 1; i < size; i++){
            MPI_Isend(grad_buffer, count*sizeof(T), MPI_CHAR, i, message_tag, communicator, &reqs[num_reqs++]);
        }
    MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(grad_buffer, count*sizeof(T), MPI_CHAR, 0, message_tag, communicator, MPI_STATUS_IGNORE);
    }
}

template <typename T, typename F>
void MsAllreduceOp::SyncLocalReduce(T *grad_buffer, T *recv_buffer, int count, MPI_Comm communicator, int message_tag, TensorTableEntry entry, F dotProdFunc)
{
    int rank;
    int size;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &size);
    MPI_Request* reqs = new MPI_Request[(size-1)*2];
    int num_reqs = 0;
    for (int i = 0; i < size; i++){
        if (i != rank){
            MPI_Irecv((void*)&recv_buffer[count/size*i], count/size*sizeof(T), MPI_CHAR, i, message_tag, communicator, &reqs[num_reqs++]);
            MPI_Isend((void*)&grad_buffer[count/size*i], count/size*sizeof(T), MPI_CHAR, i, message_tag, communicator, &reqs[num_reqs++]);
        } else {
            memcpyUtil(entry, (void *)&recv_buffer[count/size*i], (void *)&grad_buffer[count/size*i], count/size*sizeof(T));
            //memcpy(&recv_buffer[count/size*i], (void *)&grad_buffer[count/size*i], count/size*sizeof(T));
        }
    }
    MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);
    for (int i = 1; i < size; i++){
        PairwiseReduceWithComm(recv_buffer, &recv_buffer[count/size*i], count/size, message_tag, communicator, true, dotProdFunc);
    }
    num_reqs = 0;
    if (rank == 0){
        for (int i = 1; i < size; i++){
            MPI_Irecv(&grad_buffer[count/size*i], count/size*sizeof(T), MPI_CHAR, i, message_tag, communicator, &reqs[num_reqs++]);
        }
        memcpyUtil(entry, (void *)grad_buffer, (void *)recv_buffer, count/size*sizeof(T));
        //memcpy(grad_buffer, recv_buffer, count/size*sizeof(T));
        MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);
    } else {
        MPI_Send(recv_buffer, count/size*sizeof(T), MPI_CHAR, 0, message_tag, communicator);
    }
    delete[] reqs;
}

template<typename T, typename F>
void MsAllreduceOp::SyncAllreduce(T* grad_buffer, T* recv_buffer, int count, MPI_Comm communicator, MPI_Comm* reduction_comms, int message_tag, F dotProdFunc) {
    int rank;
    int size;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &size);
    //MPI_Allreduce((float*) grad_buffer, (float*) recv_buffer, count/2, MPI_FLOAT, MPI_SUM, communicator);
    //return;

    int chunk_size = (1<<15);
    int nearest_power_2 = 1;
    for (nearest_power_2 = 1; (nearest_power_2<<1) <= size; nearest_power_2 = (nearest_power_2 << 1)){}
    int remaining_non_power_2 = size - nearest_power_2;
    int level;
    if (rank >= size - 2 * remaining_non_power_2){
        int myCount;
        int nghrCount;
        level = 0;
        int neighbor_rank;
        int sendOffset;
        int recvOffset;
        if (rank < nearest_power_2){
            neighbor_rank = rank + remaining_non_power_2;
            myCount = (count >> 1);
            nghrCount = count - myCount;
            sendOffset = myCount;
            recvOffset = 0;
        } else {
            nghrCount = (count >> 1);
            myCount = count - nghrCount;
            neighbor_rank = rank - remaining_non_power_2;
            sendOffset = 0;
            recvOffset = nghrCount;
        }
        for (int i = 0; i < std::max(nghrCount, myCount); i += chunk_size) {
            MPI_Sendrecv((char*)(&grad_buffer[i+sendOffset]), std::min(chunk_size, nghrCount-i)*sizeof(T)/sizeof(char), MPI_CHAR, neighbor_rank, level * 1000 + message_tag, (char*)(&recv_buffer[i+recvOffset]), std::min(chunk_size, myCount-i)*sizeof(T)/sizeof(char), MPI_CHAR, neighbor_rank, level * 1000 + message_tag, communicator, MPI_STATUS_IGNORE);
        }
        ScaledAdd(myCount, 1.0, &grad_buffer[recvOffset] , 1.0, &recv_buffer[recvOffset]);

        if (rank < nearest_power_2) {
            for (int i = 0; i < nghrCount; i += chunk_size) {
                MPI_Recv((char*)(&grad_buffer[i+sendOffset]), std::min(chunk_size, nghrCount-i)*sizeof(T)/sizeof(char), MPI_CHAR, neighbor_rank, level * 1000 + message_tag, communicator, MPI_STATUS_IGNORE);
            }
        } else {
            for (int i = 0; i < myCount; i += chunk_size)
                MPI_Send((char*)(&grad_buffer[i+recvOffset]), std::min(chunk_size, myCount-i)*sizeof(T)/sizeof(char), MPI_CHAR, neighbor_rank, level * 1000 + message_tag, communicator);
        }
    }

    int orgSize = size;
    size = nearest_power_2;
    if (rank < nearest_power_2){
        int myCount = count;
        int comm_index;
        for (level = 1, comm_index = 0; level < size; level = (level << 1), comm_index++){
            int neighbor_rank = rank ^ level;
            int nghrCount = 0;
            int sendOffset = 0;
            int recvOffset = 0;
            int firstHalfMyCount = (myCount >> 1);
            int secondHalfMyCount = myCount - firstHalfMyCount;
            if ((rank & level) != 0) {
                myCount = secondHalfMyCount;
                nghrCount = firstHalfMyCount;
                sendOffset = 0;
                recvOffset = nghrCount;
            } else {
                myCount = firstHalfMyCount;
                nghrCount = secondHalfMyCount;
                sendOffset = myCount;
                recvOffset = 0;
            }
            for (int i = 0; i < std::max(myCount,nghrCount); i += chunk_size)
                MPI_Sendrecv((char*)(&grad_buffer[i+sendOffset]), std::min(chunk_size, nghrCount-i)*sizeof(T)/sizeof(char), MPI_CHAR, neighbor_rank, level * 1000 + message_tag, (char*)(&recv_buffer[i+recvOffset]), std::min(chunk_size, myCount-i)*sizeof(T)/sizeof(char), MPI_CHAR, neighbor_rank, level * 1000 + message_tag, communicator, MPI_STATUS_IGNORE);
            if ((rank & level) != 0) {
                grad_buffer = &grad_buffer[nghrCount];
                recv_buffer = &recv_buffer[nghrCount];
            }
            if (level == 1) {
                ScaledAdd(myCount, 0.5, grad_buffer , 0.5, recv_buffer);
            } else {
                PairwiseReduceWithComm(grad_buffer, recv_buffer, myCount, message_tag, reduction_comms[comm_index], (rank & level) == 0, dotProdFunc<T>);
            }
        }

            for (level = (size >> 1); level > 0; level = (level >> 1)) {
                int neighbor_rank = rank ^ level;
                int nghrCount = myCount;
                int levelNP = (level << 1);
                int levelSizeDeterminer = levelNP - 1;
                int countRemainer = (count & levelSizeDeterminer);
                int myLevelRank = (rank & levelSizeDeterminer);
                int nghrLevelRank = (neighbor_rank & levelSizeDeterminer);
                if ((myLevelRank >= (levelNP - countRemainer)) && (nghrLevelRank < (levelNP - countRemainer))){
                    nghrCount -= 1;
                } else if ((myLevelRank < (levelNP - countRemainer)) && (nghrLevelRank >= (levelNP - countRemainer))){
                    nghrCount += 1;
                }

                if ((rank & level) == 0) {
                    recv_buffer = &grad_buffer[myCount];
                } else {
                    recv_buffer = &grad_buffer[-nghrCount];
                }
                for (int i = 0; i < std::max(myCount,nghrCount); i += chunk_size)
                    MPI_Sendrecv((char*)(&grad_buffer[i]), std::min(chunk_size, myCount-i)*sizeof(T)/sizeof(char), MPI_CHAR, neighbor_rank, level * 1000 + message_tag, (char*)(&recv_buffer[i]), std::min(chunk_size, nghrCount-i)*sizeof(T)/sizeof(char), MPI_CHAR, neighbor_rank, level * 1000 + message_tag, communicator, MPI_STATUS_IGNORE);
                if ((rank & level) != 0) {
                    grad_buffer = &grad_buffer[-nghrCount];
                }
                myCount += nghrCount;
            }
    }
    size = orgSize;

    if (rank >= size - 2 * remaining_non_power_2){
        level = 0;
        int neighbor_rank;
        if (rank < nearest_power_2) {
            neighbor_rank = rank + remaining_non_power_2;
            for (int i = 0; i < count; i += chunk_size) {
                MPI_Send((char*)(&grad_buffer[i]), std::min(chunk_size, count-i)*sizeof(T)/sizeof(char), MPI_CHAR, neighbor_rank, level * 1000 + message_tag, communicator);
            }
        } else {
            neighbor_rank = rank - remaining_non_power_2;
            for (int i = 0; i < count; i += chunk_size)
                MPI_Recv((char*)(&grad_buffer[i]), std::min(chunk_size, count-i)*sizeof(T)/sizeof(char), MPI_CHAR, neighbor_rank, level * 1000 + message_tag, communicator, MPI_STATUS_IGNORE);
        }
    }

}
// TODO new parasail algo end
} // namespace common
} // namespace horovod