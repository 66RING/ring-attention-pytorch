import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

import os
import torch
import torch.distributed as dist
from torch.distributed import batch_isend_irecv, P2POp, isend, irecv

# Sequence parallel group that the current rank belongs to.
_SEQUENCE_PARALLEL_GROUP = None
# These values enable us to change the sequence parallel sizes on the fly.
_SEQUENCE_PARALLEL_SIZE = None
_SEQUENCE_PARALLEL_RANK = None

# setup distributed environment
def initialize_distributed(backend="nccl"):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
    else:
        if int(os.environ["RANK"]) == 0:
            print("Initializing Torch distributed.")
        dist.init_process_group(backend=backend)
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        global_world_size = dist.get_world_size()
        torch.cuda.set_device(dist.get_rank() % local_world_size)

    _initialize_sequence_parallel()
   # create_nccl_communicators()

def _initialize_sequence_parallel(sequence_parallel_size=None):
    # Get world size and rank. Ensure some consistencies.
    assert sequence_parallel_size is None, "Multiple sequence parallel group not implemented."
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if sequence_parallel_size is None:
        sequence_parallel_size = world_size
    else:
        assert world_size % sequence_parallel_size == 0
    num_sequence_parallel_groups: int = world_size // sequence_parallel_size

    rank = torch.distributed.get_rank()

    # Build the sequence parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_RANK
    global _SEQUENCE_PARALLEL_SIZE

    assert (
        _SEQUENCE_PARALLEL_GROUP is None
    ), 'sequence parallel group is already initialized'
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group
            _SEQUENCE_PARALLEL_RANK = ranks.index(rank)
            _SEQUENCE_PARALLEL_SIZE = len(ranks)

    if dist.get_rank() == 0:
        print("************ Finish sequence pralell group Initialization. ***********")


def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    #global _SEQUENCE_PARALLEL_GROUP
    assert (
        _SEQUENCE_PARALLEL_GROUP is not None
    ), 'sequence parallel group is not initialized'
    return _SEQUENCE_PARALLEL_GROUP

def get_sequence_parallel_rank():
    """Return my rank for the sequence  parallel group."""
    global _SEQUENCE_PARALLEL_RANK
    if _SEQUENCE_PARALLEL_RANK is not None:
        return _SEQUENCE_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_sequence_parallel_group())

def get_sequence_parallel_size():
    """Return my rank for the sequence  parallel group."""
    global _SEQUENCE_PARALLEL_SIZE
    if _SEQUENCE_PARALLEL_SIZE is not None:
        return _SEQUENCE_PARALLEL_SIZE
    return torch.distributed.get_world_size(group=get_sequence_parallel_group())

def step_kv_send_recv(send_k: torch.Tensor, recv_k: torch.Tensor,
                      send_v: torch.Tensor, recv_v: torch.Tensor,):
    seq_group = get_sequence_parallel_group()
    # TODO:which group to be determine and review
    seq_rank = get_sequence_parallel_rank()
    seq_world_size = get_sequence_parallel_size()

    # Handles for operations that actually need to be wait before going to the next iteration.
    # For instance, QKV sender never needs to wait -> it seems fusing these calls help scheduler; 
    all_handles = []

    # 倒序传送 0 <- 1 <- 2 <- 3 <- 0
    send_rank = (seq_rank - 1) % seq_world_size
    recv_rank = (seq_rank + 1) % seq_world_size

    # TODO: p2p 是否可以优化?

    # send and recv k, v
    all_handles.append(P2POp(op=isend, tensor=send_k, peer=send_rank % seq_world_size, group=seq_group))
    all_handles.append(P2POp(op=irecv, tensor=recv_k, peer=recv_rank % seq_world_size, group=seq_group))
    all_handles.append(P2POp(op=isend, tensor=send_v, peer=send_rank % seq_world_size, group=seq_group))
    all_handles.append(P2POp(op=irecv, tensor=recv_v, peer=recv_rank % seq_world_size, group=seq_group))

    reqs = batch_isend_irecv(all_handles)
    for req in reqs:
        req.wait()


