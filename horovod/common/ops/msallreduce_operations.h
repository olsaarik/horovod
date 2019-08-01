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

#ifndef HOROVOD_MSALLREDUCE_OPERATIONS_H
#define HOROVOD_MSALLREDUCE_OPERATIONS_H

#include <iostream>

#include "mpi.h"

#include "../common.h"
#include "../global_state.h"
#include "../mpi_context.h"
#include "p2p_operations.h"


namespace horovod {
namespace common {

class MsAllreduceOp : public PointToPointOp {
public:
  MsAllreduceOp(MPIContext* mpi_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries,
                         const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
                       const std::vector<TensorTableEntry>& entries,
                       const Response& response) const override;

protected:
  template<typename T>
  void MsAllreduce_Internal(T* gradient_buffer, T* result_buffer, int64_t buffer_length, Communicator communicator, int message_tag, int* layer_sizes, int num_layers);
  
  template<typename T, typename TACC>
  void PairwiseReduce_Internal(T* left_tensor, T* right_tensor, int cobuffer_lengthunt, int* layer_sizes, int num_layers);

  template<typename T, typename TACC>
  void ComputeDotAndNormSqrd(const T* __restrict__ a, const T* __restrict__ b, int n, TACC& dotProduct, TACC& normsq);
    
  template<typename T, typename TACC>
  void TAXPY(int n, TACC a, T* __restrict__ x, T* __restrict__ y);

  void Execute_helper(std::map<int, Status>& return_status, TensorTableEntry& entry, const Response& response, int layerid);
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_MSALLREDUCE_OPERATIONS_H
