import torch
from composer.trainer.trainer import Trainer
import logging
from composer.models import ComposerModel
from composer.utils import dist, get_device

log = logging.getLogger(__name__)
from torch.utils.data import DataLoader, Dataset

def before_training():
    log.warning(f'bigning debug init dist')
    dist.initialize_dist(get_device(None), timeout=60)

    t = torch.tensor([2], device=f'cuda:{dist.get_local_rank()}')

    torch.distributed.all_reduce(t)

    log.warning(f"bigning debug all reduce done")

    if dist.get_global_rank() == 0:
        raise RuntimeError(f"bigning raise error on rank 0")

    log.warning(f"bigning debug start 2nd all reduce on rank {dist.get_global_rank()}")
    torch.distributed.all_reduce(t)
    log.warning(f"bigning debug move t to cpu")
    a = t.tolist()
    log.warning(f"bigning debug return")


class SimpleDatasetForAuto(Dataset):

    def __init__(self, size: int = 256, feature_size: int = 1, num_classes: int = 2):
        self.size = size
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.x = None
        self.y = None

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Note: lazily generate data so it runs after Composer seeds everything, giving the same
        # dataset across multiple calls when using the same seed.
        if self.x is None:
            self.x = torch.randn(self.size, self.feature_size)
        if self.y is None:
            self.y = torch.randint(0, self.num_classes, size=(self.size,), dtype=torch.long)
        return self.x[index]

class MyModel(ComposerModel):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)

    def forward(self, x):
        pred = self.linear(x)
        return pred

    def loss(self, outputs, batch):
        return torch.sum(outputs)

def main():
    before_training()
    

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
    main()
