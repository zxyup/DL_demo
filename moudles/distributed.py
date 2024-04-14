import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

rank = int(os.environ["RANK"])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
print(f"rank:{rank}, local_rank:{local_rank}, world_size:{world_size}")

