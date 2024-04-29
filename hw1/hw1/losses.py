import abc
import torch
import torch.nn.functional as F


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        # Number of samples
        num_samples = x_scores.shape[0]

        # Get scores of the correct classes
        correct_class_scores = x_scores[torch.arange(num_samples), y].unsqueeze(1)

        # Calculate the margins for all class scores
        margins = x_scores - correct_class_scores + self.delta

        # Zero the margins for the correct class to ignore them in the loss calculation
        margins[torch.arange(num_samples), y] = 0

        # Calculate the hinge loss: max(0, margins)
        loss = F.relu(margins).sum(dim=1).mean()  # Sum over classes and average over samples

        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
<<<<<<< HEAD
        # raise NotImplementedError()
=======
        # Save context for gradient calculation
        self.grad_ctx = {'x': x, 'y': y, 'margins': margins, 'num_samples': num_samples}
>>>>>>> LNN-3600/master
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        x = self.grad_ctx['x']
        y = self.grad_ctx['y']
        margins = self.grad_ctx['margins']
        num_samples = self.grad_ctx['num_samples']

        # Create an indicator matrix where margins contribute to the loss
        positive_margins = margins > 0

        # Initialize the gradient matrix for scores
        grad_scores = torch.zeros_like(margins)
        grad_scores[positive_margins] = 1
        grad_scores[torch.arange(num_samples), y] -= positive_margins.sum(dim=1)

        # Compute the gradient with respect to the weights as x^T @ grad_scores
        grad = x.T @ grad_scores / num_samples
        # ========================

        return grad.T
