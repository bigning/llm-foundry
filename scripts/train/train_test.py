import torch
import logging
from composer.utils import dist, get_device

log = logging.getLogger(__name__)

def main():
    log.debug(f'bigning debug init dist')
    dist.initialize_dist(get_device(None), timeout=300)

    t = torch.tensor([2], device=f'cuda:{dist.get_local_rank()}')

    torch.distributed.all_reduce(t)

    log.debug(f"bigning debug all reduce done")

    if dist.get_global_rank() == 0:
        raise RuntimeError(f"bigning raise error on rank 0")

    log.debug(f"bigning debug start 2nd all reduce on rank {dist.get_global_rank()}")
    torch.distributed.all_reduce(t)


if __name__ == "__main__":
    main()
