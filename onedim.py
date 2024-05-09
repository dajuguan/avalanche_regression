import torch
import numpy as np
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks import dataset_benchmark,nc_benchmark,ni_benchmark,benchmark_from_datasets
from avalanche.benchmarks.utils import make_avalanche_dataset
from avalanche.benchmarks.utils import DataAttribute, ConstantSequence
from plugins import CustomReplay, CustomAccuracyPlugin
from avalanche.training.storage_policy import ParametricBuffer, RandomExemplarsSelectionStrategy
from avalanche.training.templates import SupervisedTemplate, BaseSGDTemplate
from avalanche.training.plugins import ReplayPlugin, EWCPlugin, EvaluationPlugin 
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    timing_metrics,
    cpu_usage_metrics,
    confusion_matrix_metrics,
    disk_usage_metrics,
    AccuracyPerTaskPluginMetric,
    Accuracy
)
from avalanche.core import SupervisedPlugin, BaseSGDPlugin

import matplotlib.pyplot as plt

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
#### gen second data
x_data_1 = torch.rand(100, 3) + 2
y_data_1 = torch.Tensor([(x_data_1[:,0]*3).tolist(), (x_data_1[:,1]*2).tolist(),(x_data_1[:,2]*1).tolist()]).transpose(0,1)
# torch.sum(x_data, dim=1)
print(y_data_1.size())

dummy_task_labels = torch.ones(100)
# for regression we have to put this because there is a the batch is: [x, y, label]
torch_data = TensorDataset(x_data_1, y_data_1,dummy_task_labels)   
torch_data.targets = [0 for _ in range(len(torch_data))]

tl = DataAttribute(ConstantSequence(1, len(torch_data)), "targets_task_labels")
avl_data_1 = AvalancheDataset([torch_data],data_attributes=[tl])

## dataloader

tl = DataAttribute(ConstantSequence(0, len(avl_data)), "targets_task_labels")
train_s = benchmark_from_datasets(train=[avl_data, avl_data_1])
print("trains:", train_s.train_stream)


## trainning
from torch.optim import SGD, Adam
from torch.nn import MSELoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC  # and many more!

model = SimpleMLP(input_size=3, num_classes=3, hidden_size=128, hidden_layers=2, drop_rate=0.0)
# optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = Adam(model.parameters(),lr=0.01, weight_decay=0.00001)
criterion = MSELoss(reduction="mean")

replay = ReplayPlugin(mem_size=300)
ewc = EWCPlugin(ewc_lambda=0.0001)

eval_plugin2 = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True, trained_experience=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    # loggers=[InteractiveLogger()]
)


cust_plugin = CustomAccuracyPlugin(model=model)

cl_strategy = SupervisedTemplate(
    model, optimizer, criterion,
    plugins=[replay, ewc, cust_plugin],
    train_epochs=40,
    evaluator=eval_plugin2,
    train_mb_size=2,
    # eval_mb_size=50
    )
# cl_strategy = Naive(
#     model, optimizer, criterion,
#     train_mb_size=10, 
#     train_epochs=5, 
#     plugins=[CustomReplay(storage_p)],
#     eval_mb_size=1
# )

# TRAINING LOOP
accuracies = np.array([])
print('Starting experiment...')
for exp_id, experience in enumerate(train_s.train_stream):
    # print("Start of task ", experience.task_label)
    # print("Start of experience ", experience.current_experience)
    print("experience._dataset._datasets.tensors", experience._dataset._datasets[0])
    cl_strategy.train(experience)
    # print('Training completed')

    print('Computing accuracy on the current test set')
    acc = cl_strategy.eval(experience)
    print("========================>", acc)

print("acc_experiences lemn:", len(cust_plugin.acc_experiences))
for i in range(0, len(cust_plugin.task_accuracies)):
    print("ploting:", i)
    data = cust_plugin.task_accuracies[i]["data"]
    plt.plot(np.linspace(0,len(data),num=len(data)) + cust_plugin.task_accuracies[i]["epoch_start"],data)
plt.savefig("loss.png")
exit(0)

print("metric_dict = eval_plugin2.get_all_metrics()=============>")
print(" eval_plugin2.get_all_metrics() keys:",  eval_plugin2.get_all_metrics().keys())

res = eval_plugin2.get_all_metrics()["Top1_Acc_Epoch/train_phase/train_stream"]
plt.plot(res[0], res[1])
# print("res===========>", res) 
plt.savefig("loss_0.png")
plt.close()

# res = eval_plugin2.get_all_metrics()["Loss_Exp/eval_phase/train_stream/Exp001"]
# print("res===========>", res)
# plt.plot(res[0], res[1])
# plt.savefig("loss_1.png")

exit(0)

x = x_data[1,:]
x = torch.Tensor([x.tolist()])
model.eval()
predict_y = model(x)
print("x:", x)
print("y          : ===========>", y_data[1,:])
print("predicted y: ===========>", predict_y[0])
print("dividen:", predict_y / x)


x = x_data_1[1,:]
x = torch.Tensor([x.tolist()])
model.eval()
predict_y = model(x)
print("x:", x)
print("y          : ===========>", y_data_1[1,:])
print("predicted y: ===========>", predict_y[0])
print("dividen:", predict_y / x)