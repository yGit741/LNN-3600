import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        # Create weights tensor of appropriate dimensions
        # Initialize it from a normal distribution with zero mean and the given std.
        self.weights = torch.randn(n_classes, n_features) * weight_std
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======

        # Implement linear prediction:
        # Calculate the score for each class using the weights
        class_scores = x.matmul(self.weights.t())  # (N, n_classes)

        # Return the class y_pred with the highest score
        y_pred = torch.argmax(class_scores, dim=1)  # (N,)

        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        if y.shape != y_pred.shape:
            raise ValueError("The shape of ground truth labels and predicted labels must match.")
        # Calculate the number of correct predictions
        correct = (y == y_pred).sum().item()  # .item() converts a tensor with one element to a Python scalar
        # Calculate accuracy as the number of correct predictions divided by the total number of predictions
        total = y.shape[0]
        acc = correct / total
        # ========================

        return acc * 100

    def _train_epoch(self, dataloader, loss_fn, learn_rate, weight_decay, res):
        total_loss, total_correct, total_samples = 0, 0, 0
        for x, y in dataloader:
            y_pred, class_scores = self.predict(x)
            loss = loss_fn.loss(x, y, class_scores, y_pred)
            loss += weight_decay * self.weights.pow(2).sum()  # Regularization

            # Gradient calculation and weight update
            self.weights -= learn_rate * (loss_fn.grad() + 2 * weight_decay * self.weights)

            total_loss += loss.item() * x.size(0)
            total_correct += (y_pred == y).sum().item()
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100
        res.loss.append(avg_loss)
        res.accuracy.append(accuracy)

    def _eval_epoch(self, dataloader, loss_fn, res):
        total_loss, total_correct, total_samples = 0, 0, 0
        for x, y in dataloader:
            y_pred, class_scores = self.predict(x)
            loss = loss_fn.loss(x, y, class_scores, y_pred)
            total_loss += loss.item() * x.size(0)
            total_correct += (y_pred == y).sum().item()
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100
        res.loss.append(avg_loss)
        res.accuracy.append(accuracy)

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            # Training loop
            self._train_epoch(dl_train, loss_fn, learn_rate, weight_decay, train_res)
            # Validation loop
            self._eval_epoch(dl_valid, loss_fn, valid_res)
            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        if has_bias:
            w_images = self.weights[:, 1:].view(self.n_classes, *img_shape)
        else:
            w_images = self.weights.view(self.n_classes, *img_shape)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp = dict(weight_std=0.05, learn_rate=0.01, weight_decay=0.0001)
    # ========================

    return hp


