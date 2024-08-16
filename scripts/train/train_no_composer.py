import torch
import torch.multiprocessing as mp
import logging
import torch.distributed as dist



log = logging.getLogger(__name__)
from torch.utils.data import DataLoader, Dataset

def before_training(local_rank):
    log.warning(f'bigning debug init dist')

    t = torch.tensor([2], device=f'local_rank')

    torch.distributed.all_reduce(t)

    log.warning(f"bigning debug all reduce done")

    if dist.get_rank() == 0:
        raise RuntimeError(f"bigning raise error on rank 0")

    log.warning(f"bigning debug start 2nd all reduce on rank {dist.get_rank()}")
    torch.distributed.all_reduce(t)
    log.warning(f"bigning debug move t to cpu")
    a = t.tolist()
    log.warning(f"bigning debug return")


def main(rank, world_size):
    ## INITIALIZE DIST
    # Running on one node so master_addr is just local host
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "28000"

    local_rank = rank

    rank = 8 * int(os.environ["NODE_RANK"]) + rank



    # All ranks simulataneously init the process group together.
    #dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=10))
    dist.init_process_group(rank=rank, world_size=world_size, timeout=timedelta(seconds=60))
 
    before_training(local_rank)
    

    """
    model = MyModel()
    dataset = SimpleDatasetForAuto(size=256, feature_size=16)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD(model.linear.parameters(), lr=0.01)

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        parallelism_config={
            'fsdp': {
                'use_orig_params': True,
            },
        },
        max_duration='3ba',
        device='gpu',
        dist_timeout=60,
        python_log_level='debug',
    )
    trainer.fit()
    """




if __name__ == "__main__":
    world_size = 16
    mp.spawn(
        main,
        args = (world_size, ),
        nprocs=8,
        join=True
    )
