import torch
import time
import math
import torch.distributed as dist
import triton
import triton.language as tl

from src.utils import *
from communication import *

from src.flash_attn_triton import flash_attn_triton
from src.utils import ref_attn
try:
    from flash_attn import flash_attn_func as flash_attn_func_official
except ImportError:
    flash_attn_func_official = None
    

# Prepare buffer for send and recv
def prepare_kv_double_buffer(k, v):
    # TODO: dose it deep copy?

    # Move k,v into buffer
    buffer_k = []
    buffer_v = []
    buffer_k.append(k)
    buffer_k.append(torch.empty_like(k))
    buffer_v.append(v)
    buffer_v.append(torch.empty_like(v))
    return buffer_k, buffer_v


def test_p2p_ring_pipeline():
    torch.manual_seed(66)
    initialize_distributed()
    debug = False

    BS, HEAD, SEQLEN, DIM = 4, 16, 128, 64
    # BS, HEAD, SEQLEN, DIM = 1, 1, 4, 1
    q, k, v = get_tensors(BS, HEAD, SEQLEN, DIM)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    seq_per_rank = SEQLEN // world_size

    full_q = q.clone()
    full_k = k.clone()
    full_v = v.clone()

    a, b, c, d = q.size()
    real_q = q[:,:, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].view(a, b, -1, d).contiguous().clone().detach().requires_grad_(True)
    real_k = k[:,:, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].view(a, b, -1, d).contiguous().clone().detach().requires_grad_(True)
    real_v = v[:,:, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].view(a, b, -1, d).contiguous().clone().detach().requires_grad_(True)


    if rank == 0:
        print("world_size", world_size)
        if debug:
            print("full_k", full_k)
            # print("full_v", full_v)

    # init double buffering
    buffer_k, buffer_v = prepare_kv_double_buffer(real_k, real_v)

    processed_k = torch.tensor([], device="cuda")
    processed_v = torch.tensor([], device="cuda")

    for time_step in range(world_size):
        # send buffer
        buf_id1 = time_step % 2
        # recv buffer
        buf_id2 = (time_step - 1) % 2
        step_kv_send_recv(buffer_k[buf_id1], buffer_k[buf_id2], buffer_v[buf_id1], buffer_v[buf_id2])
        # print(f"rank {rank} time_step {time_step} buffer_k", buffer_k[buf_id1])
        # ring: k0 -> k1 -> k2 -> k3 -> k0
        processed_k = torch.cat([processed_k, buffer_k[buf_id1]], dim=2)
        processed_v = torch.cat([processed_v, buffer_v[buf_id1]], dim=2)

    
    # NOTE: roll may take extra memory
    processed_k = torch.roll(processed_k, rank * seq_per_rank, dims=2)
    processed_v = torch.roll(processed_v, rank * seq_per_rank, dims=2)

    if debug:
        print(f"rank {rank} processed_k {processed_k}")
        # print(f"rank {rank} processed_v {processed_v}")

    assert torch.allclose(full_k, processed_k.half())
    assert torch.allclose(full_v, processed_v.half())


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, O,
    # NOTE: 开始时rank 0 got qkv 0
    MAX, DENOM,
    stride_q_bs, stride_q_head, stride_q_seqlen, stride_q_dim,
    stride_k_bs, stride_k_head, stride_k_seqlen, stride_k_dim,
    stride_v_bs, stride_v_head, stride_v_seqlen, stride_v_dim,
    # TODO: 其他stride这里是不需要的, 因为q, k, v, o的形状一样
    BS, HEAD, SEQLEN,
    DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    # BLOCK_M用于做Q的分块
    BLOCK_M: tl.constexpr,
    # BLOCK_N用于做K和V的分块
    BLOCK_N: tl.constexpr,
):
    # grid = (cdiv(seqlen, BLOCK_M), bs * head)
    # triton.language.program_id(axis) axis is The axis of the 3D launch grid
    # Q分块的起始地址
    start_m = tl.program_id(0)
    # NOTE: 计算chunk内偏移
    # start_m % SEQLEN
    # start_m = start_m % SEQLEN
    # 跳过(bs, head)的偏移
    off_bs_head = tl.program_id(1)

    qkv_base_offset = off_bs_head * stride_q_head
    Q_block_ptr = tl.make_block_ptr(
        # base offset to skip to the right (bs, head)
        base=Q + qkv_base_offset,
        # the shape of parent
        shape=(SEQLEN, DIM),
        strides=(stride_q_seqlen, stride_q_dim),
        # offset of the block inside of parent block
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        # base offset to skip to the right (bs, head)
        base=K + qkv_base_offset,
        # the shape of parent
        # NOTE: make_block_ptr读入时将K转置了
        shape=(DIM, SEQLEN),
        strides=(stride_k_dim, stride_k_seqlen),
        # 每个Q需要遍历整个的k和v
        offsets=(0, 0),
        # K根据BLOCK_N分块
        block_shape=(DIM, BLOCK_N),
        # 读入K的转置
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        # base offset to skip to the right (bs, head)
        base=V + qkv_base_offset,
        # the shape of parent
        shape=(SEQLEN, DIM),
        strides=(stride_k_seqlen, stride_v_dim),
        # 每个Q需要遍历整个的k和v
        offsets=(0, 0),
        # K根据BLOCK_N分块
        block_shape=(BLOCK_N, DIM),
        order=(1, 0),
    )
    # initialize offsets
    # NOTE: BLOCK_M表示Q的分块大小, BLOCK_N表示k, v的分块大小
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # NOTE: 一次处理一个(BLOCK_M, dim)的q, 而max和分母的sum都只需要一维, 即(BLOCK_M, 1)
    max_ptr = MAX + off_bs_head * SEQLEN + offs_m
    max = tl.load(max_ptr)
    # 分母累加的sum, 每行的sum是一样的, 所以只需要一维然后广播即可
    denom_ptr = DENOM + off_bs_head * SEQLEN + offs_m
    denom = tl.load(denom_ptr)

    # out_buffer从外界反复读入
    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_q_seqlen, stride_q_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    out_buffer = tl.load(O_block_ptr)
    out_buffer = out_buffer.to(tl.float32)
    # out_buffer = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504
    # load q: stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    lo = 0
    # NOTE:: CAUSAL就是常说的不能看到后面的文本的自回归模型
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else SEQLEN

    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        # compute qk
        # NOTE: q.shape = (BLOCK_M, dim), k.shape(已转置) = (dim, BLOCK_N)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # NOTE: 执行矩阵乘法(matrix product), k在make_block_ptr时已经转置
        # qk init as zero
        qk += tl.dot(q, k)

        # compute scaling constant

        # NOTE:
        # max.shape = [BLOCK_M], aka [BLOCK_M, 1]
        # qk.shape = [BLOCK_M, BLOCK_N]
        # tl.max(block, axis)
        # tl.maximum(block, block)
        max_new = tl.maximum(max, tl.max(qk, 1))
        # 保存exp的值, 节省exp操作
        alpha = tl.math.exp2(max - max_new)
        # NOTE:
        # nume = e^{x - max(x)}
        # max.shape = [BLOCK_M], max_new[:, None]扩展成[BLOCK_M, 1]来做广播操作
        nume = tl.math.exp2(qk - max_new[:, None])
        # scale and update acc 
        # NOTE: 利用广播来快速构建scale用于更新分母
        out_scale = denom * 0 + alpha
        # NOTE: 
        # out_scale.shape = l_i.shape = [BLOCK_M]
        # out_scale[:, None]扩展成[BLOCK_M, 1]来做广播操作
        # out_buffer = old_out * scale来更新分子
        out_buffer *= out_scale[:, None]
        out_buffer += tl.dot(nume.to(tl.float16), v)
        # update max and denominator
        denom = denom * alpha + tl.sum(nume, 1)
        max = max_new
        # update k v pointer
        # NOTE: 计算下一个k, v的分块
        # 因为k已经转置(dim, seqlen), 所以算下一批seq的k时是增加k的第二个维度
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    tl.store(max_ptr, max)
    tl.store(denom_ptr, denom)

    # write back O
    tl.store(O_block_ptr, out_buffer.to(tl.float16))

@triton.jit
def _rescale(
    L, O,
    # NOTE: 开始时rank 0 got qkv 0
    DENOM,
    stride_o_bs, stride_o_head, stride_o_seqlen, stride_o_dim,
    BS, HEAD, SEQLEN,
    DIM: tl.constexpr,
    # BLOCK_M用于做Q的分块
    BLOCK_M: tl.constexpr,
    # BLOCK_N用于做K和V的分块
    BLOCK_N: tl.constexpr,
):
    # grid = (cdiv(seqlen, BLOCK_M), bs * head)
    # triton.language.program_id(axis) axis is The axis of the 3D launch grid
    # Q分块的起始地址
    start_m = tl.program_id(0)
    # NOTE: 计算chunk内偏移
    # start_m % SEQLEN
    # start_m = start_m % SEQLEN
    # 跳过(bs, head)的偏移
    off_bs_head = tl.program_id(1)

    qkv_base_offset = off_bs_head * stride_o_head
    # initialize offsets
    # NOTE: BLOCK_M表示Q的分块大小, BLOCK_N表示k, v的分块大小
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # 分母累加的sum, 每行的sum是一样的, 所以只需要一维然后广播即可
    denom_ptr = DENOM + off_bs_head * SEQLEN + offs_m
    denom = tl.load(denom_ptr)

    # out_buffer需要加载外部的out_buffer有部分数据的out_buffer
    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_o_seqlen, stride_o_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0)
    )
    out_buffer = tl.load(O_block_ptr)
    out_buffer = out_buffer.to(tl.float16)

    # TODO: 整理并处理精度转换问题, 不用转来转去的
    out_buffer = out_buffer / denom[:, None]
    tl.store(O_block_ptr, out_buffer.to(tl.float16))

# TODO: 目前没有block wise, 直接combin
# NOTE: 一slice q只能产出一slice的o 和 L!
def ring_attention(q, k, v, causal=True, sm_scale=1):

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # shape constraints
    bs, head, seqlen, dim = q.shape
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    # 创建发送和接收的buffer
    buffer_k, buffer_v = prepare_kv_double_buffer(k, v)

    max = torch.full((bs, head, seqlen), fill_value=-float("inf"), device=q.device, dtype=torch.float32).contiguous()
    denom = torch.zeros((bs, head, seqlen), device=q.device, dtype=torch.float32).contiguous()
    local_o = torch.empty_like(q)

    BLOCK_M = 128
    BLOCK_N = 64

    group_size = triton.cdiv(seqlen, BLOCK_M)
    grid = (group_size, q.shape[0] * q.shape[1], 1)

    # NOTE: 
    # L.shape = (bs * head, seqlen)
    # L记录了所有的分母和mi(m_i + tl.math.log2(l_i)), 用于后续的backward
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    # 设置适当的wrap以提升性能
    num_warps = 4 if Lk <= 64 else 8
    for time_step in range(world_size):
        # send buffer
        buf_id1 = time_step % 2
        # recv buffer
        buf_id2 = (time_step - 1) % 2

        local_q = q
        local_k = buffer_k[buf_id1]
        local_v = buffer_v[buf_id1]

        # TODO: gpu async?
        _fwd_kernel[grid](
            local_q, local_k, local_v, sm_scale,
            L, local_o,
            max, denom,
            local_q.stride(0), local_q.stride(1), local_q.stride(2), local_q.stride(3),
            local_k.stride(0), local_k.stride(1), local_k.stride(2), local_k.stride(3),
            local_v.stride(0), local_v.stride(1), local_v.stride(2), local_v.stride(3),
            bs, head, seqlen,
            DIM=Lk,
            IS_CAUSAL=causal,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, 
            num_warps=num_warps,
            num_stages=4)

        # TODO: 必须吗
        # TODO: 但用户态调用sync 其实会很耗费时间?
        torch.cuda.synchronize()
        step_kv_send_recv(buffer_k[buf_id1], buffer_k[buf_id2], buffer_v[buf_id1], buffer_v[buf_id2])

    # TODO: 多次kernel启动加上中间的多次类型转换会导致严重的精度丢失

    _rescale[grid](
            L, local_o,
            denom,
            local_o.stride(0), local_o.stride(1), local_o.stride(2), local_o.stride(3),
            bs, head, seqlen,
            DIM=Lk,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.cuda.synchronize()


    # TODO: 没有blockwise, 直接gather
    res_o = [torch.empty_like(q, dtype=local_o.dtype) for _ in range(world_size)]
    dist.all_gather(res_o, local_o)
    res_o = torch.cat(res_o, dim=-2)

    return res_o


def test_ring_attention():
    torch.manual_seed(66)
    initialize_distributed()

    BS, HEAD, SEQLEN, DIM = 4, 16, 128 * 4, 64
    # BS, HEAD, SEQLEN, DIM = 1, 1, 4, 1
    q, k, v = get_tensors(BS, HEAD, SEQLEN, DIM)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    seq_per_rank = SEQLEN // world_size

    full_q = q.clone()
    full_k = k.clone()
    full_v = v.clone()

    a, b, c, d = q.size()
    real_q = q[:,:, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].view(a, b, -1, d).contiguous().clone().detach().requires_grad_(True)
    real_k = k[:,:, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].view(a, b, -1, d).contiguous().clone().detach().requires_grad_(True)
    real_v = v[:,:, rank * seq_per_rank: (rank + 1) * seq_per_rank, :].view(a, b, -1, d).contiguous().clone().detach().requires_grad_(True)

    sm_scale = 1.0
    is_causal = False
    ring_out = ring_attention(real_q, real_k, real_v, is_causal, sm_scale)
    ref_out = ref_attn(full_q, full_k, full_v, is_causal, sm_scale)

    assert torch.allclose(ref_out[:,:,: seq_per_rank,:], ring_out[:,:,:seq_per_rank,:], atol=1e-2, rtol=0)


def test_op():
    pass

def main():
    # test_p2p_ring_pipeline()
    test_ring_attention()


if __name__ == "__main__":
    main()

