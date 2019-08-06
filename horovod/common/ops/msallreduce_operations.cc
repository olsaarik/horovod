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
    [&return_statuses, this, &e, response, layerid]
    {
      LOG(INFO, global_state_->rank)<<"Begin processing tensor in layer "<<layerid;
      Execute_helper(return_statuses, e, response, layerid);
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

void MsAllreduceOp::Execute_helper(std::map<int, Status>& return_status, TensorTableEntry& entry, const Response& response, int layerid){
  void* buffer_data;
  int buffer_len;
  void* recv_buffer;

  buffer_data = (void*) entry.tensor->data();

  buffer_len = entry.output->size();

  FusionBufferManager buffer_manager;
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
  switch (entry.output->dtype()) {
    case HOROVOD_INT8:
      //TODO new parasail
    SyncAllreduce((int8_t*) buffer_data,
                    (int8_t*) recv_buffer,
                    buffer_len,
                    Communicator::GLOBAL,
                    global_state_->reduction_comms,
                    layerid);  
    // TODO old parasail 
    // MsAllreduce_Internal((int8_t*) buffer_data,
    //                     (int8_t*) recv_buffer,
    //                     buffer_len,
    //                     Communicator::GLOBAL,
    //                     layerid,
    //                     &buffer_len,
    //                     1);
    break;     
    case HOROVOD_UINT8:
    //TODO new parasail
    SyncAllreduce((uint8_t*) buffer_data,
                    (uint8_t*) recv_buffer,
                    buffer_len,
                    Communicator::GLOBAL,
                    global_state_->reduction_comms,
                    layerid);  
    // TODO old parasail
    // MsAllreduce_Internal((uint8_t*) buffer_data,
    //                     (uint8_t*) recv_buffer,
    //                     buffer_len,
    //                     Communicator::GLOBAL,
    //                     layerid,
    //                     &buffer_len,
    //                     1);
    break;
    case HOROVOD_FLOAT16:
    //TODO new parasail
    SyncAllreduce((MsAllreduceOp::float16*) buffer_data,
                    (MsAllreduceOp::float16*) recv_buffer,
                    buffer_len,
                    Communicator::GLOBAL,
                    global_state_->reduction_comms,
                    layerid);  
    break;
    case HOROVOD_UINT16:
    //TODO new parasail
    SyncAllreduce((uint16_t*) buffer_data,
                    (uint16_t*) recv_buffer,
                    buffer_len,
                    Communicator::GLOBAL,
                    global_state_->reduction_comms,
                    layerid);  
    // TODO old parasail
    // MsAllreduce_Internal((uint16_t*) buffer_data,
    //                     (uint16_t*) recv_buffer,
    //                     buffer_len,
    //                     Communicator::GLOBAL,
    //                     layerid,
    //                     &buffer_len,
    //                     1);
    break;
    case HOROVOD_INT16:
    //TODO new parasail
    SyncAllreduce((int16_t*) buffer_data,
                    (int16_t*) recv_buffer,
                    buffer_len,
                    Communicator::GLOBAL,
                    global_state_->reduction_comms,
                    layerid);  
    // TODO old parasail
    // MsAllreduce_Internal((int16_t*) buffer_data,
    //                     (int16_t*) recv_buffer,
    //                     buffer_len,
    //                     Communicator::GLOBAL,
    //                     layerid,
    //                     &buffer_len,
    //                     1);
    break;
    case HOROVOD_INT32:
    //TODO new parasail
    SyncAllreduce((int32_t*) buffer_data,
                    (int32_t*) recv_buffer,
                    buffer_len,
                    Communicator::GLOBAL,
                    global_state_->reduction_comms,
                    layerid);  
    // TODO old parasail
    // MsAllreduce_Internal((int32_t*) buffer_data,
    //                     (int32_t*) recv_buffer,
    //                     buffer_len,
    //                     Communicator::GLOBAL,
    //                     layerid,
    //                     &buffer_len,
    //                     1);
    break;
    case HOROVOD_INT64:
    //TODO new parasail
    SyncAllreduce((int64_t*) buffer_data,
                    (int64_t*) recv_buffer,
                    buffer_len,
                    Communicator::GLOBAL,
                    global_state_->reduction_comms,
                    layerid);
    // TODO old parasail
    // MsAllreduce_Internal((int64_t*) buffer_data,
    //                     (int64_t*) recv_buffer,
    //                     buffer_len,
    //                     Communicator::GLOBAL,
    //                     layerid,
    //                     &buffer_len,
    //                     1);
    break;
    case HOROVOD_FLOAT32:
    //TODO new parasail
    SyncAllreduce((float*) buffer_data,
                    (float*) recv_buffer,
                    buffer_len,
                    Communicator::GLOBAL,
                    global_state_->reduction_comms,
                    layerid);
    // TODO old parasail
    // MsAllreduce_Internal((float*) buffer_data,
    //                     (float*) recv_buffer,
    //                     buffer_len,
    //                     Communicator::GLOBAL,
    //                     layerid,
    //                     &buffer_len,
    //                     1);
    break;
    case HOROVOD_FLOAT64:
    //TODO new parasail
    SyncAllreduce((double*) buffer_data,
                    (double*) recv_buffer,
                    buffer_len,
                    Communicator::GLOBAL,
                    global_state_->reduction_comms,
                    layerid);
    
    // TODO old parasail
    // MsAllreduce_Internal((double*) buffer_data,
    //                     (double*) recv_buffer,
    //                     buffer_len,
    //                     Communicator::GLOBAL,
    //                     layerid,
    //                     &buffer_len,
    //                     1);
    break;
    default:
        throw std::logic_error("MsAllreduceOp::Execute_helper: UNsupported data type.");
  }
  if(entry.tensor->data() == entry.output->data()) {
    // Return the buffer back into the pool of available buffers
    global_state_->buffer_lock.lock();
    global_state_->temp_buffers.push(buffer_manager);
    global_state_->buffer_lock.unlock();
  }
  std::memcpy((void*)entry.output->data(), buffer_data,
                (size_t)entry.tensor->size());

  LOG(INFO, global_state_->rank)<<"Finished ms allreduction, exiting operation";
}

bool MsAllreduceOp::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

// TODO new parasail algo begin
template<typename T>
void MsAllreduceOp::ComputeDotAndNormSqrds(const T* __restrict__  a, const T* __restrict__ b, int n, double& dotProduct, double& anormsq, double& bnormsq) {
    dotProduct = 0.;
    anormsq = 0.;
    bnormsq = 0.;

    for (int i = 0; i < n; i++) {
        dotProduct += a[i] * b[i];
        anormsq += a[i] * a[i];
        bnormsq += b[i] * b[i];
    }
}

template<typename T>
void MsAllreduceOp::ScaledAdd(int n, double acoeff, T* __restrict__ a, double bcoeff, T* __restrict__ b) {
    for (int i = 0; i < n; i++) {
        a[i] = acoeff * a[i] + bcoeff * b[i];
    }
}

template<typename T>
void MsAllreduceOp::PairwiseReduceWithComm(T* a, T* b, int count, int message_tag, MPI_Comm& comm, bool isLeftNeighbor) {
    double dotProduct = 0.f;
    double anormsq = 0.f;
    double bnormsq = 0.f;
    ComputeDotAndNormSqrds(a, b, count, dotProduct, anormsq, bnormsq);

    double reduce_vals[3];
    double recv_reduce_vals[3];
            if (isLeftNeighbor) { 
                reduce_vals[0] = anormsq;
                reduce_vals[1] = bnormsq;
            } else {
                reduce_vals[1] = anormsq;
                reduce_vals[0] = bnormsq;
            }
    reduce_vals[2] = dotProduct;

    MPI_Allreduce(MPI_IN_PLACE, reduce_vals, 3, MPI_DOUBLE, MPI_SUM, comm);

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
}

template<typename T>
void MsAllreduceOp::SyncAllreduce(T* grad_buffer, T* recv_buffer, int count, Communicator common_comm, MPI_Comm* reduction_comms, int message_tag) {
    int rank;
    int size;
    MPI_Comm communicator = mpi_context_->GetMPICommunicator(common_comm);
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &size);
    count = count / sizeof(T);
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
                T* org_grad_buffer = grad_buffer;
                T* org_recv_buffer = recv_buffer;
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
                        LOG(INFO,global_state_->rank)<<"comm_index is"<<comm_index;
                        PairwiseReduceWithComm(grad_buffer, recv_buffer, myCount, message_tag, reduction_comms[comm_index], (rank & level) == 0);
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

template<typename T>
void MsAllreduceOp::MsAllreduce_Internal(T* gradient_buffer, T* result_buffer, int64_t buffer_length, Communicator communicator, int message_tag, int* layer_sizes, int num_layers){
    int true_rank;
    int redn_rank;
    int size;
    MPI_Comm_rank(mpi_context_->GetMPICommunicator(communicator), &true_rank);
    MPI_Comm_size(mpi_context_->GetMPICommunicator(communicator), &size);
    LOG(INFO, global_state_->rank)<<"Starting ms allreduction internal";

    int root_node_rotation = message_tag % size;
    redn_rank = (true_rank + root_node_rotation) % size;

    // Do a tree reduction
    // The reduction ranks used are a permutation of true ranks (permuted based on message_tag) 
    // This spreads the load of tree reduction across different true ranks

    // at each level l, node X0[0..0] receives from X1[0...], 
    // where [0..0] is l zeros in the bit representation of the rank of a node

    int level;
    for (level = 1; level < size; level *= 2) {
        int neighbor_redn_rank = redn_rank ^ level;
        int neighbor_true_rank = (neighbor_redn_rank + size - root_node_rotation) % size;
        if (redn_rank % level != 0)
            continue; // stay idle at this level

        if (neighbor_redn_rank >= size)
            continue; // no neighbor and so stay idle at this level

        if ((redn_rank & level) == 0) {
            // recv buffer from neighbor
            LOG(INFO, global_state_->rank)<<std::this_thread::get_id()<<" Reduction: receiving from neighbor";
            PointToPointRecv(result_buffer, buffer_length, neighbor_true_rank, message_tag, communicator);

            PairwiseReduce_Internal<T, T>(gradient_buffer, result_buffer, (int) buffer_length, layer_sizes, num_layers);
        }
        else {
            // send gradient_buffer to neighbor
            LOG(INFO, global_state_->rank)<<std::this_thread::get_id()<<" Reduction: sending to neighbor";
            PointToPointSend(gradient_buffer, buffer_length, neighbor_true_rank, message_tag, communicator);
        }
    }

    // Do a inverse tree to do a broadcast
    // cannot use MPI Broadcast as there can be concurrent Allreduces happening in parallel

    // the same logic as above. 
    // at each level l, node X0[0..0] sends to X1[0...], 
    // where [0..0] is l zeros in the bit representation of the rank of a node

    level /= 2; // this make sure that level < size

    for (; level > 0; level /= 2) {
        int neighbor_redn_rank = redn_rank ^ level;
        int neighbor_true_rank = (neighbor_redn_rank + size - root_node_rotation) % size;

        if (redn_rank % level != 0)
            continue; // stay idle at this level

        if (neighbor_redn_rank >= size)
            continue; // no neighbor and so stay idle at this level

        if ((redn_rank & level) == 0) {
            // send gradient_buffer to neighbor
            // and dont wait for the send to finish
            LOG(INFO, global_state_->rank)<<std::this_thread::get_id()<<" Reverse bcasting: sending to neighbor";
            PointToPointSend(gradient_buffer, buffer_length, neighbor_true_rank, message_tag, communicator);
        }
        else {
            // recv gradient_buffer from 
            LOG(INFO, global_state_->rank)<<std::this_thread::get_id()<<" Reverse bcasting: Receiving from neighbor";
            PointToPointRecv(gradient_buffer, buffer_length, neighbor_true_rank, message_tag, communicator);
        }
    }
    LOG(INFO, global_state_->rank)<<std::this_thread::get_id()<<" Exiting msallreduction_internal.";
}
template<typename T, typename TACC>
void MsAllreduceOp::PairwiseReduce_Internal(T* left_tensor, T* right_tensor, int buffer_length, int* layer_sizes, int num_layers){
    LOG(INFO, global_state_->rank)<<"Starting pairwise reduction internal";
    //TODO make this multi-threaded
    int nt = omp_get_max_threads();
    //int nt = 1;
    
    // Get number of elements
    int count = buffer_length/sizeof(T);
    for(int i = 0; i < num_layers; i++) {
        layer_sizes[i] = layer_sizes[i]/sizeof(T);
        LOG(INFO, global_state_->rank)<<"layer_sizes at i: "<<i<<" is "<<layer_sizes[i];
    }
    LOG(INFO, global_state_->rank)<<"Processing total "<<count<<" elements";

    const int cache_alignment = 64 / sizeof(TACC);

    std::vector<TACC> norms(num_layers*cache_alignment, (TACC)0);
    std::vector<TACC> dotProducts(num_layers*cache_alignment, (TACC)0);

    // reduction is parallelized uniformly across available OpenMP threads
#pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        int numThreads = omp_get_num_threads();
        int myStart = (count * tid) / numThreads;
        int myEnd = (count * (tid + 1)) / numThreads;

        // go over each layer and process the layer only if it overlaps with [myStart, myEnd)
        for (int i = 0, layer_begin = 0; i < num_layers; layer_begin += layer_sizes[i++]) {
            int layer_end = layer_begin + layer_sizes[i];

            if (myEnd <= layer_begin)
                break; // no overlap now and in future

            if (myStart >= layer_end)
                continue; // no overlap yet

            int begin = std::max(myStart, layer_begin);
            int end = std::min(myEnd, layer_end);

            // compute dotProduct of a[begin, end) and b[begin, end) together with norm square of b[begin, end)
            TACC locDotProduct = 0.f;
            TACC locNorm = 0.f;
            ComputeDotAndNormSqrd(&left_tensor[begin], &right_tensor[begin], end - begin, locDotProduct, locNorm);
#pragma omp critical
            {
                // multiplied by cache_alignment to avoid false sharing
                dotProducts[i * cache_alignment] += locDotProduct;
                norms[i * cache_alignment] += locNorm;
            }
        }
#pragma omp barrier

        // go over each layer and process the layer only if it overlaps with [myStart, myEnd)
        for (int i = 0, layer_begin = 0; i < num_layers; layer_begin += layer_sizes[i++]) {
            int layer_end = layer_begin + layer_sizes[i];

            if (myEnd <= layer_begin)
                break; // no overlap now and in future

            if (myStart >= layer_end)
                continue; // no overlap yet

            TACC dotProduct = dotProducts[i*cache_alignment];
            TACC norm = norms[i*cache_alignment];

            TACC thresh = 1e-18f;
            TACC coeff = 0.0f;
            if (std::abs(norm) < thresh)
                coeff = 1.0f;
            else
                coeff = 1.f - dotProduct / norm;

            int begin = std::max(myStart, layer_begin);
            int end = std::min(myEnd, layer_end);

            // a[begin, end) += coeff * b[begin, end)
            // where coeff = 1 - a.b/normbsq
            TAXPY(end - begin, coeff, &right_tensor[begin], &left_tensor[begin]);
        }
    }
}

template<typename T, typename TACC>
void MsAllreduceOp::ComputeDotAndNormSqrd(const T* __restrict__ a, const T* __restrict__ b, int n, TACC& dotProduct, TACC& normsq){
    dotProduct = 0.;
    normsq = 0.;

    for (int i = 0; i < n; i++) {
        dotProduct += a[i] * b[i];
        normsq += b[i] * b[i];
    }
}

template<typename T, typename TACC>
void MsAllreduceOp::TAXPY(int n, TACC a, T* __restrict__ x, T* __restrict__ y){
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

} // namespace common
} // namespace horovod