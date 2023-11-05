## torch API

- `apply()`
    * [解答](https://discuss.pytorch.org/t/difference-between-apply-an-call-for-an-autograd-function/13845)
    * 新的函数回调方式, 只能定义在`@staticmethod`中
    * TODO, 目的
- `torch.manual_seed(seed)`控制随机值方便测试!!
- TODO `repeat`


- 魔改的flashattention是否正确?
    * 应该正确, 参考自openai的工程师, 但和原版flashattn有所区别

- 需要注意!! lightseq执行时, q, k, v就已经ddp分配好了!
    * **lightseq的测试值得一看**

## triton API

- [`tl.max(block, axis)`](https://triton-lang.org/main/python-api/generated/triton.language.max.html#triton-language-max)
- [`tl.maximum(block, block)`](https://triton-lang.org/main/python-api/generated/triton.language.maximum.html)
- [`tl.sum`](https://triton-lang.org/main/python-api/generated/triton.language.sum.html)
- `triton.cdiv(x, y)`, c for ceil
- `tl.make_block_ptr()`
    * base: the base pointer to the parent tensor
    * shape: the shape of the parent tensor
        + API会根据order指定的顺序取构造所需的shape
    * strides: 步长, 当api将指针增加1时会自动结合步操作
    * offsets: the offsets of the block
    * block_shape: the shape of the block
    * order: the order of the block, which means how the block is laid out in memory
        + e.g. (1, 0), which means the second axis is the inner dimension in terms of storage
- `tl.where(condition, x, y)`
    * 相当于`condition ? x : y`


```python
a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                order=(1, 0))
```


## IMPORTANT NOTE

- [triton impl of flash attention2](https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py)


