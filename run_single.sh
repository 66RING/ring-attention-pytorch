
python -m torch.distributed.run --nproc_per_node=1 \
  --nnodes=1 --node_rank=0\
  ./ring_attention.py
