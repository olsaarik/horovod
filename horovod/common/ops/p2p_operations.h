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

#ifndef HOROVOD_P2P_OPERATIONS_H
#define HOROVOD_P2P_OPERATIONS_H

#include <iostream>

#include "../common.h"
#include "../global_state.h"
#include "collective_operations.h"


namespace horovod {
namespace common {

class PointToPointOp : public AllreduceOp {
public:
  PointToPointOp(HorovodGlobalState* global_state);

  virtual ~PointToPointOp() = default;

protected:
//TODO provide interfaces for other communication libraries
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_P2P_OPERATIONS_H
