from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate, BaseSGDTemplate
from avalanche.evaluation.metrics import Accuracy


class CustomAccuracyPlugin(SupervisedPlugin):
    def __init__(self, model):
        """ A simple replay plugin with reservoir sampling. """
        super().__init__()
        self.model = model
        self.task_accuracies = [{"epoch_start":0, "data":[]}]
        self.epoch = 0
        self.acc_experiences = []

    def before_training(self, strategy: SupervisedTemplate, *args, **kwargs):
        """Called before `train` by the `BaseTemplate`."""

        # Save model's initial weights in the first experience training step
        return

    def before_training_exp(self, strategy: SupervisedTemplate, *args, **kwargs):
        """ Use a custom dataloader to combine samples from the current data and memory buffer. """
        self.acc_experiences.append(strategy.experience)

    def after_training_exp(self, strategy: SupervisedTemplate, **kwargs):
        """ Update the buffer. """
        print("updating experiences............")

    def after_backward(self, strategy: SupervisedTemplate, *args, **kwargs):
        """Called after `criterion.backward()` by the `BaseTemplate`."""
        pass

    def after_training_epoch(self, strategy: SupervisedTemplate, *args, **kwargs):
        """Called after `train_epoch` by the `BaseTemplate`."""
        pass

    def after_training_iteration(self, strategy: SupervisedTemplate, *args, **kwargs):
        """Called after the end of a training iteration by the
        `BaseTemplate`."""
        # model = self.model
        model = strategy.model
        model.eval()
        for i in range(len(self.acc_experiences)):
            experience = self.acc_experiences[i]
            data = experience._dataset._datasets[0].tensors
            x_data = data[0]
            y_data = data[1]
            acc_metric = Accuracy()
            acc_metric.update(y_data, model(x_data))
            if  i == len(self.task_accuracies):
                self.task_accuracies.append({"epoch_start":self.epoch, "data":[]})
            self.task_accuracies[i]["data"].append(acc_metric.result())
        model.train()
        self.epoch += 1

class CustomReplay(SupervisedPlugin):
    def __init__(self, storage_policy):
        super().__init__()
        self.storage_policy = storage_policy

    def before_training_exp(self, strategy,
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """ Here we set the dataloader. """
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        # replay dataloader samples mini-batches from the memory and current
        # data separately and combines them together.
        print("Override the dataloader.")
        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle)

    def after_training_exp(self, strategy: "BaseStrategy", **kwargs):
        """ We update the buffer after the experience.
            You can use a different callback to update the buffer in a different place
        """
        print("Buffer update.")
        self.storage_policy.update(strategy, **kwargs)
