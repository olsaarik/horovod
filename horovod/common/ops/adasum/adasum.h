// Copyright 2019 Microsoft. All Rights Reserved.
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

#ifndef HOROVOD_ADASUM_H
#define HOROVOD_ADASUM_H

#include <cstring>
#include <stack>
#include <immintrin.h>
#include <emmintrin.h>

#include "../../common.h"
#include "../../global_state.h"


namespace horovod {
namespace common {

static bool IsPowerOfTwo(ulong x)
{
  return (x != 0) && ((x & (x - 1)) == 0);
}

// Interface for Adasum algorithm
template <typename Communicator_type>
class Adasum {
public:
  Adasum(HorovodGlobalState* global_state) {
    // Allocate receive buffer size equal to the fusion buffer length
    current_recv_buffer_length = global_state->parameter_manager.TensorFusionThresholdBytes();
    recv_buffer_ = (uint8_t*)malloc(current_recv_buffer_length);
  };

  ~Adasum() {
    if(recv_buffer_ != nullptr) {
      free(recv_buffer_);
    }
  }
protected:
  // Temp buffer used by Adasum operations
  uint8_t* recv_buffer_ = nullptr;

  // Get recv buffer
  virtual uint8_t* GetRecvBuffer(int buffer_length) {
    return CheckBufferAndReallocate(&recv_buffer_, buffer_length, current_recv_buffer_length);
  } 
  
  // Check buffer length and re-allocate if necessary
  virtual uint8_t* CheckBufferAndReallocate(uint8_t** buffer, uint64_t buffer_length, uint64_t &current_length) {
    if(buffer_length <= current_length) {
      return *buffer;
    }
    *buffer = (uint8_t*)realloc(*buffer, buffer_length);
    current_length = buffer_length;
    return *buffer;
  }

  // Communication primitives required for Adasum algorithm
  virtual void PointToPointSendRecv(void* input_data_buffer,
                                    int64_t input_buffer_length,
                                    void* output_data_buffer,
                                    int64_t output_buffer_length,
                                    DataType horovod_datatype,
                                    int dst_src_rank,
                                    int tag,
                                    Communicator_type communicator,
                                    HorovodGlobalState* global_state) = 0;

  virtual int GetLocalRankWithComm(Communicator_type communicator) = 0;

  virtual int GetSizeWithComm(Communicator_type communicator) = 0;

  int GetPerElementSize(TensorTableEntry entry) {
   return GetPerElementSize(entry.tensor->dtype());
  }

  int GetPerElementSize(DataType horovod_datatype) {
    switch(horovod_datatype) {
        case DataType::HOROVOD_FLOAT16:
          return 2;
        case DataType::HOROVOD_FLOAT32:
          return 4;
        case DataType::HOROVOD_FLOAT64:
          return 8;
        default:
          throw std::logic_error("Unsupported data type.");
    }
  }

  virtual void DispatchComputeDotAndNormSqrds(const void* __restrict__  a,
                                              const void* __restrict__ b,
                                              DataType horovod_datatype,
                                              int count,
                                              double& dotProduct,
                                              double& anormsq,
                                              double& bnormsq,
                                              int layerid) {
    if (horovod_datatype == DataType::HOROVOD_FLOAT16) {
      ComputeDotAndNormSqrdsfp16((uint16_t*)a, (uint16_t*)b, count, dotProduct, anormsq, bnormsq, layerid);
    }
    else if (horovod_datatype == DataType::HOROVOD_FLOAT32) {
      ComputeDotAndNormSqrds((float*)a, (float*)b, count, dotProduct, anormsq, bnormsq, layerid);
    }
    else if (horovod_datatype == DataType::HOROVOD_FLOAT64) {
      ComputeDotAndNormSqrds((double*)a, (double*)b, count, dotProduct, anormsq, bnormsq, layerid);      
    }
    else {
        throw std::logic_error("Unsupported data type.");
    }
  }
  
  // Given two vectors compute their dot product and the squared norm for each.
  template<typename T>
  void ComputeDotAndNormSqrds(const T* __restrict__  a, const T* __restrict__ b, int count, double& dotProduct, double& anormsq, double& bnormsq, int layerid) {
      dotProduct = 0.;
      anormsq = 0.;
      bnormsq = 0.;

      for (int i = 0; i < count; i++) {
          dotProduct += a[i] * b[i];
          anormsq += a[i] * a[i];
          bnormsq += b[i] * b[i];
      }
  }
  
  virtual void DispatchScaledAdd(DataType horovod_datatype,
                                 int count,
                                 double acoeff,
                                 void* __restrict__ a,
                                 double bcoeff,
                                 void* __restrict__ b,
                                 int layerid){
    if (horovod_datatype == DataType::HOROVOD_FLOAT16) {
      ScaledAddfp16(count, acoeff, (uint16_t*)a, bcoeff, (uint16_t*)b, layerid);
    }
    else if (horovod_datatype == DataType::HOROVOD_FLOAT32) {
      ScaledAdd(count, acoeff, (float*)a, bcoeff, (float*)b, layerid);
    }
    else if (horovod_datatype == DataType::HOROVOD_FLOAT64) {
      ScaledAdd(count, acoeff, (double*)a, bcoeff, (double*)b, layerid);
    }
    else {
        throw std::logic_error("Unsupported data type.");
    }
  }
  
  // Update a vector to a linear combination of itself and another vector.
  template<typename T>
  void ScaledAdd(int n, double acoeff, T* __restrict__ a, double bcoeff, T* __restrict__ b, int layerid) {
    for (int i = 0; i < n; i++) {
        a[i] = acoeff * a[i] + bcoeff * b[i];
    }
  }
  
  // Perform Adasum allreduce using a vector-halving, distance-doubling (VHDD) approach.
  // grad_buffer: holds the data to reduce and will hold the result.
  // recv_buffer: must point to a buffer of the same size as grad_buffer.
  // horovod_datatype: the element type of grad_buffer.
  // tensor_counts: is a list of how many elements grad_buffer contains for each tensor
  //                involved in the allreduce. It should contain a 0 if this rank holds
  //                no data for the tensor (see ranks_per_tensor below for when this can
  //                happen).
  // ranks_per_tensor: how many ranks are tensors initially shared across. Set to 1 if
  //                   each rank holds unreduced data. When set to n>1 the first log_2(n)
  //                   levels of VHDD are skipped. This is useful when the communication 
  //                   inside the node is implemented using another mechanism, e.g. NCCL's
  //                   reduce-scatter and allgather operations before and after, respectively.
  //                   When ranks_per_tensor>1, tensor_counts must be set according to the
  //                   slices owned by this rank.
  // communicator: the communicator to reduce with.
  // tag: a value used as the message tag for each send/recv in this algorithm. This is
  //      useful for multithreaded scenarios. Remember to also create separate
  //      reduction_comms instances when running with multiple threads.
  // reduction_comms: pointer to an array of communicators for computing dot products and
  //                  norms for Adasum. The communicators should include exactly the ranks
  //                  that this rank has either directly or indirectly communicated with
  //                  after each level of VHDD.
  template <typename T>
  void FusedAllreduce(std::vector<TensorTableEntry>& entries, 
                      T* grad_buffer,
                      T* recv_buffer,
                      DataType horovod_datatype,
                      std::vector<int> tensor_counts,
                      int ranks_per_tensor,
                      Communicator_type communicator,
                      int tag,
                      Communicator_type* reduction_comms,
                      HorovodGlobalState *global_state) {
    int per_element_size = GetPerElementSize(horovod_datatype);
    int rank = GetLocalRankWithComm(communicator);
    int size = GetSizeWithComm(communicator);

    if (IsPowerOfTwo(size) == false) {
      throw std::logic_error(
          "Running Adasum with non-power-of-two ranks is not supported yet.");
    }

    // Stack of total number of elements owned by the neighbor at the corresponding
    // level of the algorithm. This stack is built during the first phase of the
    // algorithm and is unwound during the second phase of the algorithm.
    std::stack<int> nghrCounts;

    // Buffer used for the distributed dot product and squared norm computations.
    std::vector<double> normAndDots(tensor_counts.size()*3);

    // Total number of elements owned by this rank
    int myCount = 0;
    for (int i = 0; i < tensor_counts.size(); i++) {
      myCount += tensor_counts[i];
    }

    // First phase of VHDD. On each level this rank is paired with a neighbor,
    // with to which it sends one half of its elements and from which it receives
    // and reduces the other half (vector-halving). On each level the distance to
    // the neighbor doubles (distance-doubling).
    int distance;
    int comm_index;
    for (distance = 1, comm_index = 0; distance < size;
        distance = (distance << 1), comm_index++) {
      if (distance < ranks_per_tensor) {
        continue;
      }

      int neighbor_rank = rank ^ distance;
      int nghrCount = 0;
      int sendOffset = 0;
      int recvOffset = 0;
      int firstHalfMyCount = (myCount >> 1);
      int secondHalfMyCount = myCount - firstHalfMyCount;
      bool isLeftNeighbor = (rank & distance) == 0;

      // If we are the left neighbor we send the second half of our data and if we are
      // the right neighbor we send the first half. Also Update how many elements of
      // each tensor this rank will hold after this level.
      if (isLeftNeighbor) {
        myCount = firstHalfMyCount;
        nghrCount = secondHalfMyCount;
        sendOffset = myCount;
        recvOffset = 0;
        
        // We lose elements from the end
        int nghrCountSoFar = 0;
        for (int i = tensor_counts.size() - 1; i >= 0; --i) {
          if (nghrCountSoFar + tensor_counts[i] <= nghrCount){
            nghrCountSoFar += tensor_counts[i];
            tensor_counts[i] = 0;
          } else {
            tensor_counts[i] -= (nghrCount - nghrCountSoFar);
            break;
          }
        }
      } else {
        myCount = secondHalfMyCount;
        nghrCount = firstHalfMyCount;
        sendOffset = 0;
        recvOffset = nghrCount;

        // We lose elements from the beginning
        int nghrCountSoFar = 0;
        for (int i = 0; i < tensor_counts.size(); ++i) {
          if (nghrCountSoFar + tensor_counts[i] <= nghrCount){
            nghrCountSoFar += tensor_counts[i];
            tensor_counts[i] = 0;
          } else {
            tensor_counts[i] -= (nghrCount - nghrCountSoFar);
            break;
          }
        }
      }

      // Remember how many elements we are giving to our neighbor
      nghrCounts.emplace(nghrCount);

      // Exchange my half with neighbor's half
      this->PointToPointSendRecv((char*)(&grad_buffer[sendOffset]),
                                 nghrCount * per_element_size,
                                 (char*)(&recv_buffer[recvOffset]),
                                 myCount * per_element_size,
                                 horovod_datatype,
                                 neighbor_rank,
                                 tag,
                                 communicator,
                                 global_state);
      
      // If we gave away our first half adjust the starting point
      if (!isLeftNeighbor) {
        grad_buffer = &grad_buffer[nghrCount];
        recv_buffer = &recv_buffer[nghrCount];
      }

      // Apply the Adasum operation inside the current reduction group
      FusedPairwiseReduceWithComm(entries,
                                  (uint8_t*)grad_buffer,
                                  (uint8_t*)recv_buffer,
                                  horovod_datatype,
                                  tensor_counts,
                                  tag,
                                  reduction_comms[comm_index],
                                  isLeftNeighbor,
                                  normAndDots,
                                  global_state);
    }

    // Second phase of VHDD. This is essentially an all gather operation.
    // On each level this rank sends all of its elements to its neighbor and
    // receives all of its neighbors elements. The neighbors are those of
    // the first phase in reverse.
    for (distance = (size >> 1); distance > 0; distance = (distance >> 1)) {
      if (distance < ranks_per_tensor){
        continue;
      }
      int neighbor_rank = rank ^ distance;
      bool isLeftNeighbor = (rank & distance) == 0;

      // Receive as many elements from our neighbor as we sent in the first phase
      int nghrCount = nghrCounts.top();
      nghrCounts.pop();

      // Receive into the same position as we sent data from
      if (isLeftNeighbor) {
        recv_buffer = &grad_buffer[-nghrCount];
      } else {
        recv_buffer = &grad_buffer[myCount];
      }
      this->PointToPointSendRecv(grad_buffer,
                                 myCount * per_element_size,
                                 recv_buffer,
                                 nghrCount * per_element_size,
                                 horovod_datatype,
                                 neighbor_rank,
                                 tag,
                                 communicator,
                                 global_state);
      myCount += nghrCount;

      // If we received a new first half adjust the starting point
      if (!isLeftNeighbor) {
        grad_buffer = &grad_buffer[-nghrCount];
      }
    }
  }

  // Applies the Adasum operation to reduce two fused sets of tensors split
  // across multiple ranks. The communicator passed to this function contains
  // exactly the ranks that the tensors are split across.
  void FusedPairwiseReduceWithComm(std::vector<TensorTableEntry>& entries,
                                   uint8_t* a,
                                   uint8_t* b,
                                   DataType horovod_datatype,
                                   std::vector<int>& tensor_counts,
                                   int layerid,
                                   Communicator_type& comm,
                                   bool isLeftNeighbor,
                                   std::vector<double>& normAndDots,
                                   HorovodGlobalState *global_state) {
    int per_element_size = GetPerElementSize(horovod_datatype);

    // Compute the dot product and squared norms using the elements of the tensors
    // held on this rank.
    int bytesSoFar = 0;
    for (size_t i = 0; i < tensor_counts.size(); i++){
      double dotProduct = 0.;
      double anormsq = 0.;
      double bnormsq = 0.;

      DispatchComputeDotAndNormSqrds(&a[bytesSoFar], &b[bytesSoFar], horovod_datatype, tensor_counts[i], dotProduct, anormsq, bnormsq, layerid);
      normAndDots[i*3] = dotProduct;
      normAndDots[i*3+1] = isLeftNeighbor ? anormsq : bnormsq;
      normAndDots[i*3+2] = isLeftNeighbor ? bnormsq : anormsq;
      bytesSoFar += tensor_counts[i] * per_element_size;
    }

    // Sum all local dot products and squared norms to produce the true values.
    SumAllreduceWithComm(entries, (void*)normAndDots.data(), 3*tensor_counts.size(), DataType::HOROVOD_FLOAT64, comm, global_state);

    // Apply the Adasum operation to each tensor
    bytesSoFar = 0;
    for (size_t i = 0; i < tensor_counts.size(); i++){
      double dotProduct = normAndDots[i*3];
      double anormsq = normAndDots[i*3+(isLeftNeighbor ? 1 : 2)];
      double bnormsq = normAndDots[i*3+(isLeftNeighbor ? 2 : 1)];

      double acoeff = 1;
      double bcoeff = 1;
      if (anormsq >= 1e-8)
        acoeff = 1.0 - dotProduct / anormsq * 0.5;
      if (bnormsq >= 1e-8)
        bcoeff = 1.0 - dotProduct / bnormsq * 0.5;

      // If a and b are orthogonal, then dotProduct is zero and this is a sum.
      // If a and b are parallel, then dotProduct is |a|*|b| and this ends up as an average.
      DispatchScaledAdd(horovod_datatype, tensor_counts[i], acoeff, &a[bytesSoFar], bcoeff, &b[bytesSoFar], layerid);
      bytesSoFar += tensor_counts[i] * per_element_size;
    }
  }

  void DispatchFusedAllreduce(std::vector<TensorTableEntry>& entries, 
                              void* grad_buffer,
                              void* recv_buffer,
                              std::vector<int>& tensor_counts,
                              int ranks_per_tensor,
                              Communicator_type communicator,
                              int tag,
                              Communicator_type* reduction_comms,
                              DataType data_type,
                              HorovodGlobalState *global_state) {
      switch(data_type) {
          case DataType::HOROVOD_FLOAT16:
            FusedAllreduce(entries, (uint16_t*)grad_buffer, (uint16_t*)recv_buffer, data_type, tensor_counts, ranks_per_tensor, communicator, tag, reduction_comms, global_state);
            break;
          case DataType::HOROVOD_FLOAT32:
            FusedAllreduce(entries, (float*)grad_buffer, (float*)recv_buffer, data_type, tensor_counts, ranks_per_tensor, communicator, tag, reduction_comms, global_state);
            break;
          case DataType::HOROVOD_FLOAT64:
            FusedAllreduce(entries, (double*)grad_buffer, (double*)recv_buffer, data_type, tensor_counts, ranks_per_tensor, communicator, tag, reduction_comms, global_state);
            break;
          default:
            throw std::logic_error("Unsupported data type");
      }
  }

  // over-write ComputeDotAndNormSqrds for float16
  inline void ComputeDotAndNormSqrdsfp16(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, int len, double& dotProduct, double& anormsq, double& bnormsq, int layerid) {
      int i;
      __m256d dotProductVec = _mm256_setzero_pd();
      __m256d anormVec = _mm256_setzero_pd();
      __m256d bnormVec = _mm256_setzero_pd();
      for (i = 0; i < len - 7; i += 8) {
          __m256 aVec = MmLoaduPh(&a[i]);
          __m256 bVec = MmLoaduPh(&b[i]);
          __m256d aBot = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 0));
          __m256d aTop = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 1));
          __m256d bBot = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 0));
          __m256d bTop = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 1));
          dotProductVec = _mm256_fmadd_pd(aBot, bBot, dotProductVec);
          dotProductVec = _mm256_fmadd_pd(aTop, bTop, dotProductVec);
          anormVec = _mm256_fmadd_pd(aBot, aBot, anormVec);
          anormVec = _mm256_fmadd_pd(aTop, aTop, anormVec);
          bnormVec = _mm256_fmadd_pd(bBot, bBot, bnormVec);
          bnormVec = _mm256_fmadd_pd(bTop, bTop, bnormVec);
      }
      if (i < len) {
          __m256 aVec = MmLoaduPhPartial(&a[i], len - i);
          __m256 bVec = MmLoaduPhPartial(&b[i], len - i);
        __m256d aBot = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 0));
        __m256d aTop = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 1));
        __m256d bBot = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 0));
        __m256d bTop = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 1));
          dotProductVec = _mm256_fmadd_pd(aBot, bBot, dotProductVec);
          dotProductVec = _mm256_fmadd_pd(aTop, bTop, dotProductVec);
          anormVec = _mm256_fmadd_pd(aBot, aBot, anormVec);
          anormVec = _mm256_fmadd_pd(aTop, aTop, anormVec);
          bnormVec = _mm256_fmadd_pd(bBot, bBot, bnormVec);
          bnormVec = _mm256_fmadd_pd(bTop, bTop, bnormVec);
      }
  
      dotProduct = Mm256ReductionPd(dotProductVec);
      anormsq = Mm256ReductionPd(anormVec);
      bnormsq = Mm256ReductionPd(bnormVec);
  }

  inline void ScaledAddfp16(int len, double acoeff, uint16_t* __restrict__ a, double bcoeff, uint16_t* __restrict__ b, int layerid) {
      int i;
      __m256 acoeffVec = _mm256_set1_ps((float)(acoeff));
      __m256 bcoeffVec = _mm256_set1_ps((float)bcoeff);
      for (i = 0; i < len - 7; i += 8) {
          __m256 aVec = MmLoaduPh(&a[i]);
          __m256 bVec = MmLoaduPh(&b[i]);
          aVec = _mm256_mul_ps(acoeffVec, aVec);
          MmStorePh(&a[i], _mm256_fmadd_ps(bcoeffVec, bVec, aVec));
      }
      if (i < len) {
          __m256 aVec = MmLoaduPhPartial(&a[i], len - i);
          __m256 bVec = MmLoaduPhPartial(&b[i], len - i);
          aVec = _mm256_mul_ps(acoeffVec, aVec);
          MmStorePhPartial(&a[i], _mm256_fmadd_ps(bcoeffVec, bVec, aVec), len - i);
      }
  }

private:
  // Keep track of current recv buffer length
  uint64_t current_recv_buffer_length;

  // reduce 4xfloat64 into one double
  inline double Mm256ReductionPd(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128
    
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar      
  }

  // load 8 float16s from a and return the __m256 register
  inline __m256 MmLoaduPh(const uint16_t* a) {
      __m128i r = _mm_loadu_si128((__m128i*)(a));
      return _mm256_cvtph_ps(r);
  }

  // store 8 float16 from val into a 
  inline void MmStorePh(uint16_t* a, __m256 val) {
      __m128i r = _mm256_cvtps_ph(val, 0);
      _mm_storeu_si128((__m128i*)a, r);
  }

  // load len (< 8) float16s from a, fill the rest with 0s, and return the __m256 register
  inline __m256 MmLoaduPhPartial(const uint16_t* a, int len) {
      short e[8];
      std::memset(e, 0, sizeof(e));
      std::memcpy(e, a, std::min(len, 8) * sizeof(short));
      __m128i es = _mm_set_epi16(e[7], e[6], e[5], e[4], e[3], e[2], e[1], e[0]);
      return _mm256_cvtph_ps(es);
  }

  // store the first len (< 8) float16s from val and store into a
  inline void MmStorePhPartial(uint16_t* a, __m256 val, int len) {
      __m128i r = _mm256_cvtps_ph(val, 0);
      //for (int i = 0; i < std::min(len, 8); i++) 
      //    a[i].value = _mm_extract_epi16(r, i);
      // but we cannot do this because the second argument to _mm_extract_epi16 has to be a compile time constant 
      if (0 < len) a[0] = (short)_mm_extract_epi16(r, 0);
      if (1 < len) a[1] = (short)_mm_extract_epi16(r, 1);
      if (2 < len) a[2] = (short)_mm_extract_epi16(r, 2);
      if (3 < len) a[3] = (short)_mm_extract_epi16(r, 3);
      if (4 < len) a[4] = (short)_mm_extract_epi16(r, 4);
      if (5 < len) a[5] = (short)_mm_extract_epi16(r, 5);
      if (6 < len) a[6] = (short)_mm_extract_epi16(r, 6);
      if (7 < len) a[7] = (short)_mm_extract_epi16(r, 7);
  }
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_ADASUM_H
