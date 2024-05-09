import torch
from avalanche.evaluation.metrics import Accuracy, TaskAwareAccuracy

# create an instance of the standalone TaskAwareAccuracy metric
# initial accuracy is 0 for each task
acc_metric = TaskAwareAccuracy()
print("Initial Accuracy: ", acc_metric.result()) #  output {}

# metric updates for 2 different tasks
task_label = 0
real_y = torch.tensor([1, 2]).long()
predicted_y = torch.tensor([1, 0]).float()
acc_metric.update(real_y, predicted_y, task_label)
acc = acc_metric.result()
print("Average Accuracy: ", acc) # output 0.5 for task 0

task_label = 1
predicted_y = torch.tensor([1,2]).float() 
acc_metric.update(real_y, predicted_y, task_label)
acc = acc_metric.result() 
print("Average Accuracy: ", acc) # output 0.75 for task 0 and 1.0 for task 1

task_label = 0
predicted_y = torch.tensor([1,2]).float()
acc_metric.update(real_y, predicted_y, task_label)
acc = acc_metric.result()
print("Average Accuracy: ", acc) # output 0.75 for task 0 and 1.0 for task 1

# reset accuracy
acc_metric.reset()
print("After reset: ", acc_metric.result()) # output {}