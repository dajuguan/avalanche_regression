from avalanche.benchmarks.classic import CORe50, SplitTinyImageNet, SplitCIFAR10, \
    SplitCIFAR100, SplitCIFAR110, SplitMNIST, RotatedMNIST, PermutedMNIST, SplitCUB200
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    AccuracyPerTaskPluginMetric,
    loss_metrics)
# creating the benchmark (scenario object)
perm_mnist = PermutedMNIST(
    n_experiences=33,
    seed=1234,
    return_task_id=True
)

# recovering the train and test streams
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream



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

from avalanche.models import SimpleMLP
from avalanche.benchmarks import SplitMNIST
from avalanche.training.supervised import Naive
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from avalanche.training.plugins import ReplayPlugin, EWCPlugin, EvaluationPlugin 


# Benchmark creation
benchmark = SplitMNIST(n_experiences=5)

# Model Creation
model = SimpleMLP(num_classes=benchmark.n_classes)

eval_plugin2 = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True, trained_experience=True),
    EpochAccuracyPerTask(),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
)
# Create the Strategy Instance (MyStrategy)
cl_strategy = Naive(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(),
    evaluator=eval_plugin2
)

# iterating over the train stream
for experience in train_stream:
    print("Start of task ", experience.task_label)
    if experience.task_label == 2:
        break
    print('Classes in this task:', experience.classes_in_this_experience)

    # The current Pytorch training set can be easily recovered through the 
    # experience
    current_training_set = experience.dataset
    # ...as well as the task_label
    print('Task {}'.format(experience.task_label))
    print('This task contains', len(current_training_set), 'training examples')

    # we can recover the corresponding test experience in the test stream
    current_test_set = test_stream[experience.current_experience].dataset
    print('This task contains', len(current_test_set), 'test examples')
    cl_strategy.train(experience)



import matplotlib.pyplot as plt
print(eval_plugin2.get_all_metrics().keys())
res = eval_plugin2.get_all_metrics()["Top1_Acc_Epoch/train_phase/train_stream/Task000"]
plt.plot(res[0], res[1])
# # res = eval_plugin2.get_all_metrics()["Top1_Acc_Exp/eval_phase/train_stream/Exp001"]
# # plt.plot(res[0], res[1])
plt.savefig("loss_task_0.png")
plt.close()
res = eval_plugin2.get_all_metrics()["Top1_Acc_Epoch/train_phase/train_stream/Task001"]
plt.plot(res[0], res[1])
# # res = eval_plugin2.get_all_metrics()["Top1_Acc_Exp/eval_phase/train_stream/Exp001"]
# # plt.plot(res[0], res[1])
plt.savefig("loss_task_1.png")