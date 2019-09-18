#include "pslops.h"

namespace horovod {
namespace common {
  int PSL_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm ) {

    int redn_rank, true_rank;
    int size;
    MPI_Comm_rank(comm, &true_rank);
    MPI_Comm_size(comm, &size);
    
    int root_node_rotation = root;
    redn_rank = (true_rank ^ root_node_rotation);
    int level;
    for (level = 1; level < size; level *= 2);
    level /= 2; // this make sure that level < size
    
    for(; level > 0; level /= 2) {
      int neighbor_redn_rank = redn_rank ^ level;
      int neighbor_true_rank = (neighbor_redn_rank ^ root_node_rotation);
      if (redn_rank % level != 0)
	continue;
      if (neighbor_redn_rank >= size)
	continue;
      if ((redn_rank & level) == 0) {
	// send grad_buffer to neighbor
	// and dont wait for the send to finish
	MPI_Send(buffer, count, datatype, neighbor_true_rank, 0, comm);
      }
      else {
	// recv grad_buffer from neighbor
	MPI_Recv(buffer, count, datatype, neighbor_true_rank, 0, comm, MPI_STATUS_IGNORE);
      }
    }
    return MPI_SUCCESS;
  }
} // namespace common
} // namespace horovod
