import torch
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks import dataset_benchmark,nc_benchmark,ni_benchmark,benchmark_from_datasets
from avalanche.benchmarks.utils import make_avalanche_dataset
from avalanche.benchmarks.utils import DataAttribute, ConstantSequence

# Create a dataset of 100 data points described by 22 features + 1 class label
x_data = torch.randn(100, 3)
y_data = torch.sum(x_data, dim=1)
dummy_task_labels = torch.zeros(100)
# for regression we have to put this because there is a the batch is: [x, y, label]
torch_data = TensorDataset(x_data, y_data,dummy_task_labels)   
torch_data.targets = [0 for _ in range(len(torch_data))]

tl = DataAttribute(ConstantSequence(0, len(torch_data)), "targets_task_labels")
avl_data = AvalancheDataset([torch_data],data_attributes=[tl])
    
print(avl_data[0])
# exit(0)

## dataloader

tl = DataAttribute(ConstantSequence(0, len(avl_data)), "targets_task_labels")
train_s = benchmark_from_datasets(train=[avl_data])
print("trains:", train_s.train_stream)


## trainning
from torch.optim import SGD
from torch.nn import MSELoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC  # and many more!

model = SimpleMLP(input_size=3, num_classes=3)
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = MSELoss()
cl_strategy = Naive(
    model, optimizer, criterion,
    train_mb_size=1, 
    train_epochs=4, 
    eval_mb_size=4
)

# TRAINING LOOP
print('Starting experiment...')
for exp_id, experience in enumerate(train_s.train_stream):
    print("Start of experience ", experience.current_experience)
    print("experience: ", experience.__dict__)
    cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the current test set')
    cl_strategy.eval(train_s[exp_id])