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

#include "mpi_p2p_operations.h"

namespace horovod {
namespace common {

MPIPointToPointOp::MPIPointToPointOp(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : PointToPointOp(global_state), mpi_context_(mpi_context) {}

bool MPIPointToPointOp::Enabled(const ParameterManager& param_manager,
                                const std::vector<TensorTableEntry>& entries,
                                const Response& response) const {
  return true;
}
} // namespace common
} // namespace horovod