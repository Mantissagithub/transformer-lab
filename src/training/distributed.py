import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class DistEnv:
    is_dist: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(force_disable: bool = False) -> DistEnv:
    """Initialize torch.distributed if launched via torchrun (WORLD_SIZE > 1).

    Returns a DistEnv even on a single process; check `.is_dist` to branch.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if force_disable or world_size == 1:
        return DistEnv(
            is_dist=False,
            rank=0,
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            world_size=1,
            backend="none",
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    return DistEnv(
        is_dist=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
    )


def _mp_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None or name == "fp32":
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unknown mixed_precision: {name}")


def wrap_model(
    model: nn.Module,
    env: DistEnv,
    strategy: str = "ddp",
    fsdp_sharding: str = "full_shard",
    fsdp_mixed_precision: Optional[str] = None,
) -> nn.Module:
    if not env.is_dist:
        return model

    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP

        device_ids = [env.local_rank] if torch.cuda.is_available() else None
        return DDP(model, device_ids=device_ids)

    if strategy == "fsdp":
        from functools import partial

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        from src.model.blocks import CausalBlock

        sharding_map = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
        }
        sharding = sharding_map[fsdp_sharding]
        dtype = _mp_dtype(fsdp_mixed_precision)
        mp_policy = (
            None
            if dtype is None
            else MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        )
        wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={CausalBlock})
        return FSDP(
            model,
            sharding_strategy=sharding,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mp_policy,
            device_id=env.local_rank if torch.cuda.is_available() else None,
        )

    raise ValueError(f"unknown distributed strategy: {strategy}")
