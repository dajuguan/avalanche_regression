import torch
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks import dataset_benchmark,nc_benchmark,ni_benchmark,benchmark_from_datasets
from avalanche.benchmarks.utils import make_avalanche_dataset


def dummy_tensor_dataset():
    """Returns a PyTorch image dataset of 10 classes."""
    x = torch.rand(32, 10)
    y = torch.rand(32, 10)
    return TensorDataset(x, y)

def test_benchmark_from_datasets():
    d1 = AvalancheDataset(dummy_tensor_dataset())
    d2 = AvalancheDataset(dummy_tensor_dataset())
    d3 = AvalancheDataset(dummy_tensor_dataset())
    d4 = AvalancheDataset(dummy_tensor_dataset())

    bm = benchmark_from_datasets(train=[d1, d2], test=[d3, d4])

    # train stream
    train_s = bm.streams["train"]
    assert len(train_s) == 2
    for eid, (exp, d_orig) in enumerate(zip(train_s, [d1, d2])):
        assert exp.current_experience == eid
        for ii, (x, y) in enumerate(exp.dataset):
            torch.testing.assert_close(x, d_orig[ii][0])
            torch.testing.assert_close(y, d_orig[ii][1])

test_benchmark_from_datasets()