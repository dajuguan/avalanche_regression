import torch
import numpy as np
from torch.utils.data.dataset import TensorDataset
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks import dataset_benchmark,nc_benchmark,ni_benchmark,benchmark_from_datasets
from avalanche.benchmarks.utils import make_avalanche_dataset
from avalanche.benchmarks.utils import DataAttribute, ConstantSequence
from replay_plugin import CustomReplay
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

model = SimpleMLP(input_size=3, num_classes=3, hidden_size=128, hidden_layers=3, drop_rate=0.0)
# optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = Adam(model.parameters(),lr=0.01, weight_decay=0.00001)
criterion = MSELoss(reduction="mean")

replay = ReplayPlugin(mem_size=300)
ewc = EWCPlugin(ewc_lambda=0.0001)

class CustomAccuracyPlugin(SupervisedPlugin):

    def __init__(self):
        """ A simple replay plugin with reservoir sampling. """
        super().__init__()
        self.task1 = []
        self.task2 = []

    def before_training(self, strategy: SupervisedTemplate, *args, **kwargs):
        """Called before `train` by the `BaseTemplate`."""

        # Save model's initial weights in the first experience training step
        return

    def before_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        """ Use a custom dataloader to combine samples from the current data and memory buffer. """
        return 

    def after_training_exp(self, strategy: SupervisedTemplate, **kwargs):
        """ Update the buffer. """
        print("updating experiences............")

    def after_training_iteration(self, strategy: SupervisedTemplate, *args, **kwargs):
        """Called after the end of a training iteration by the
        `BaseTemplate`."""
        model.eval()
        acc_metric = Accuracy()
        acc_metric.update(y_data, model(x_data))
        self.task1.append(acc_metric.result())
        acc_metric = Accuracy()
        acc_metric.update(y_data_1, model(x_data_1))
        self.task2.append(acc_metric.result())
        model.train()

class EpochAccuracyPerTask(AccuracyPerTaskPluginMetric):
    """
    The average accuracy over a single training epoch.
    This plugin metric only works at training time.

    The accuracy will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochAccuracy metric.
        """

        super(EpochAccuracyPerTask, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="train"
        )

    def __str__(self):
        return "EpochAccuracyPerTask"

eval_plugin2 = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True, trained_experience=True),
    # EpochAccuracyPerTask(),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    # loggers=[InteractiveLogger()]
)


cust_plugin = CustomAccuracyPlugin()

cl_strategy = SupervisedTemplate(
    model, optimizer, criterion,
    plugins=[replay, ewc, cust_plugin],
    train_epochs=10,
    evaluator=eval_plugin2
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
    print("experience: ", experience.__dict__)
    cl_strategy.train(experience)
    # print('Training completed')

    print('Computing accuracy on the current test set')
    acc = cl_strategy.eval(experience)
    print("========================>", acc)

plt.plot(cust_plugin.task1)
plt.plot(cust_plugin.task2)
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