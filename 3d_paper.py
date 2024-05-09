import torch
import numpy as np
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks import dataset_benchmark,nc_benchmark,ni_benchmark,benchmark_from_datasets
from avalanche.benchmarks.utils import make_avalanche_dataset
from avalanche.benchmarks.utils import DataAttribute, ConstantSequence
from plugins import CustomReplay, CustomAccuracyPlugin

# Create a dataset of 100 data points described by 22 features + 1 class label
data = np.loadtxt(r"./data.csv", delimiter=",",skiprows=1, dtype=np.float32)
x_data = torch.from_numpy(data[:,5:13])
y_data = torch.from_numpy(data[:,13:-1])
NUM = len(x_data)
# torch.sum(x_data, dim=1)

dummy_task_labels = torch.zeros(NUM)
# for regression we have to put this because there is a the batch is: [x, y, label]
torch_data = TensorDataset(x_data, y_data,dummy_task_labels)   
torch_data.targets = [0 for _ in range(len(torch_data))]

tl = DataAttribute(ConstantSequence(0, len(torch_data)), "targets_task_labels")
avl_data = AvalancheDataset([torch_data],data_attributes=[tl])
    
print(x_data.shape[1], y_data.shape)

## dataloader

tl = DataAttribute(ConstantSequence(0, len(avl_data)), "targets_task_labels")
train_s = benchmark_from_datasets(train=[avl_data, avl_data])
print("trains:", train_s.train_stream)


## trainning
from torch.optim import SGD, Adam
from torch.nn import MSELoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC  # and many more!

model = SimpleMLP(input_size=x_data.shape[1], num_classes=y_data.shape[1], hidden_size=128, hidden_layers=3, drop_rate=0.0)
# optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = Adam(model.parameters(),lr=0.01, weight_decay=0.00001)
criterion = MSELoss(reduction="mean")

cl_strategy = Naive(
    model, optimizer, criterion,
    train_mb_size=10, 
    train_epochs=1, 
    eval_mb_size=1
)

# TRAINING LOOP
print('Starting experiment...')
for exp_id, experience in enumerate(train_s.train_stream):
    print("Start of experience ", experience.current_experience)
    print("experience: ", experience.__dict__)
    cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the current test set')
    cl_strategy.eval(experience)

x = x_data[1,:]
x = torch.Tensor([x.tolist()])
model.eval()
predict_y = model(x)
print("x:", x)
print("original y: =============>", y_data[1,:])
print("predicted y: ===========>", predict_y)