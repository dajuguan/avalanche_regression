import torch
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks import dataset_benchmark,nc_benchmark,ni_benchmark,benchmark_from_datasets
from avalanche.benchmarks.utils import make_avalanche_dataset
from avalanche.benchmarks.utils import DataAttribute, ConstantSequence
from replay_plugin import CustomReplay
from avalanche.training.storage_policy import ParametricBuffer, RandomExemplarsSelectionStrategy
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins import ReplayPlugin, EWCPlugin

storage_p = ParametricBuffer(
    max_size=300,
    # groupby='class',
    selection_strategy=RandomExemplarsSelectionStrategy()
)

# Create a dataset of 100 data points described by 22 features + 1 class label
N=100
x_data = torch.rand(100, 3)
y_data = torch.Tensor([(x_data[:,0]*2).tolist(), (x_data[:,1]*3).tolist(),(x_data[:,2]*4).tolist()]).transpose(0,1)
# torch.sum(x_data, dim=1)
print(y_data.size())

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
from torch.optim import SGD, Adam
from torch.nn import MSELoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC  # and many more!

model = SimpleMLP(input_size=3, num_classes=3, hidden_size=128, hidden_layers=1, drop_rate=0.0)
# optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = Adam(model.parameters(),lr=0.01, weight_decay=0.00001)
criterion = MSELoss(reduction="mean")

replay = ReplayPlugin(mem_size=100)
ewc = EWCPlugin(ewc_lambda=0.0001)
cl_strategy = SupervisedTemplate(
    model, optimizer, criterion,
    plugins=[replay, ewc])
# cl_strategy = Naive(
#     model, optimizer, criterion,
#     train_mb_size=10, 
#     train_epochs=5, 
#     plugins=[CustomReplay(storage_p)],
#     eval_mb_size=1
# )

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
print("y          : ===========>", y_data[1,:])
print("predicted y: ===========>", predict_y[0])
print("dividen:", predict_y / x)