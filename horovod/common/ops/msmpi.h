// TODO HEADER

#ifndef HOROVOD_MSMPI_H
#define HOROVOD_MSMPI_H

#include <iostream>
#include <cstring>
#include <immintrin.h>
#include <emmintrin.h>

#include "mpi.h"

#include "../common.h"
#include "../global_state.h"
#include "../mpi_context.h"


#ifdef MPI_FRIENDLY_ASSERT
#include <iostream>
#define psl_assert(expression) (void)((!!(expression)) ||		\
				      (std::cout << "psl_assert(" << #expression << ") failed at " << __FILE__ << ":" << (unsigned)(__LINE__)) << std::endl)
#else
#include <assert.h>
#define psl_assert assert
#endif

namespace horovod {
namespace common {
  // Wrapper around MPI_recvs
  // Chunks if necessary. Also optionally checks a checksum. 
  template<class T>
    static void PointToPointRecv(T* buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status) {
    psl_assert(count * (int)sizeof(T) < (1 << 31)); // otherwise fix our logic of sending MPI_CHARs

#ifdef DEBUG_SEND_RECV
    std::cout << "to receive from " << source << "," << tag << std::endl;
#endif

#ifndef MESSAGE_CHUNK_SIZE
    MPI_Recv(buf, count * sizeof(T), MPI_CHAR, source, tag, comm, status);
#else
    const int CHUNK_SIZE = MESSAGE_CHUNK_SIZE / sizeof(T);
    for (int chk = 0, cnt = 0; chk < count; chk += CHUNK_SIZE, cnt++) {
      MPI_Recv(&buf[chk], std::min(count - chk, CHUNK_SIZE) * sizeof(T), MPI_CHAR, source, tag, comm, MPI_STATUS_IGNORE);
    }
#endif

#ifdef DEBUG_SEND_RECV
    std::cout << "received from " << source << "," << tag << std::endl;
#endif

#ifdef ADD_CHECKSUM_TO_MPI_MESSAGES
    size_t checksum = ComputeChecksum(buf, count * sizeof(T));
    size_t sender_checksum;
    MPI_Recv(&sender_checksum, sizeof(size_t), MPI_CHAR, source, tag, comm, MPI_STATUS_IGNORE);
    if (sender_checksum != checksum)
      psl_assert(sender_checksum == checksum);
#endif
  }

  // Wrapper around MPI_Send
  // Chunks if necessary. Also optionally sends a checksum. 
  template<class T>
    static void PointToPointSend(const T* buf, int count, int dest, int tag, MPI_Comm comm) {
    psl_assert(count * (int)sizeof(T) < (1 << 31)); // otherwise fix our logic of sending MPI_CHARs

#ifdef DEBUG_SEND_RECV
    std::cout << "send to " << dest << "," << tag << std::endl;
#endif

#ifndef MESSAGE_CHUNK_SIZE
    MPI_Send(buf, count * sizeof(T), MPI_CHAR, dest, tag, comm);
#else
    const int CHUNK_SIZE = MESSAGE_CHUNK_SIZE / sizeof(T);
    for (int chk = 0, cnt = 0; chk < count; chk += CHUNK_SIZE, cnt++) {
      MPI_Send(&buf[chk], std::min(count - chk, CHUNK_SIZE) * sizeof(T), MPI_CHAR, dest, tag, comm);
    }
#endif

#ifdef ADD_CHECKSUM_TO_MPI_MESSAGES
    size_t checksum = ComputeChecksum(buf, count * sizeof(T));
    MPI_Send(&checksum, sizeof(size_t), MPI_CHAR, dest, tag, comm);
#endif
  }

  // Wrapper around MPI_Isend
  // Chunks if necessary. Also optionally sends a checksum. 
  template<class T>
    static void AsyncPointToPointSend(const T* buf, int count, int dest, int tag, MPI_Comm comm) {
    psl_assert(count * (int)sizeof(T) < (1 << 31)); // otherwise fix our logic of sending MPI_CHARs

#ifdef DEBUG_SEND_RECV
    std::cout << "send to " << dest << "," << tag << std::endl;
#endif

#ifndef MESSAGE_CHUNK_SIZE
    MPI_Request req;
    MPI_Isend(buf, count * sizeof(T), MPI_CHAR, dest, tag, comm, &req);
    MPI_Request_free(&req);
#else
    const int CHUNK_SIZE = MESSAGE_CHUNK_SIZE / sizeof(T);
    for (int chk = 0, cnt = 0; chk < count; chk += CHUNK_SIZE, cnt++) {
      MPI_Request req;
      MPI_Isend(&buf[chk], std::min(count - chk, CHUNK_SIZE) * sizeof(T), MPI_CHAR, dest, tag, comm, &req);
      MPI_Request_free(&req);
    }
#endif

#ifdef ADD_CHECKSUM_TO_MPI_MESSAGES
    size_t checksum = ComputeChecksum(buf, count * sizeof(T));
    MPI_Send(&checksum, sizeof(size_t), MPI_CHAR, dest, tag, comm);
#endif
  }
	      
  template<typename T>
  void MSMPI_Allreduce(T *grad_buffer, T *recv_buffer, int count, MPI_Comm communicator, int message_tag) {
    int true_rank;
    int redn_rank;
    int size;
    MPI_Comm_rank(communicator, &true_rank);
    MPI_Comm_size(communicator, &size);

    static bool opt_permute_roots_on_allreduce = false;
    int root_node_rotation = opt_permute_roots_on_allreduce ? (message_tag % size) : 0;
    redn_rank = (true_rank ^ root_node_rotation);

    psl_assert(!(grad_buffer <= recv_buffer && recv_buffer < grad_buffer + count) || (recv_buffer <= grad_buffer && grad_buffer < recv_buffer + count));

    // Do a tree reduction
    // The reduction ranks used are a permutation of true ranks (permuted based on message_tag)
    // This spreads the load of tree reduction across different true ranks

    // at each level l, node X0[0..0] receives from X1[0...],
    // where [0..0] is l zeros in the bit representation of the rank of a node

    int level;
    for (level = 1; level < size; level *= 2) {
      int neighbor_redn_rank = redn_rank ^ level;
      int neighbor_true_rank = (neighbor_redn_rank ^ root_node_rotation);
      if (redn_rank % level != 0)
	continue; // stay idle at this level

      if (neighbor_redn_rank >= size)
	continue; // no neighbor and so stay idle at this level

      if ((redn_rank & level) == 0) {
	// recv buffer from neighbor
	PointToPointRecv(recv_buffer, count, neighbor_true_rank, message_tag, communicator, MPI_STATUS_IGNORE);

	// do reduction
	for(int i = 0; i < count; i++) {
	  grad_buffer[i] += recv_buffer[i];
	}
      }
      else {
	// send grad_buffer to neighbor
	PointToPointSend(grad_buffer, count, neighbor_true_rank, message_tag, communicator);
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
      int neighbor_true_rank = (neighbor_redn_rank ^ root_node_rotation);

      if (redn_rank % level != 0)
	continue; // stay idle at this level

      if (neighbor_redn_rank >= size)
	continue; // no neighbor and so stay idle at this level

      if ((redn_rank & level) == 0) {
	// send grad_buffer to neighbor
	// and dont wait for the send to finish
	PointToPointSend(grad_buffer, count, neighbor_true_rank, message_tag, communicator);
      }
      else {
	// recv grad_buffer from neighbor
	PointToPointRecv(grad_buffer, count, neighbor_true_rank, message_tag, communicator, MPI_STATUS_IGNORE);
      }
    }
  }
  
}// namespace common
} // namespace horovod

#endif
