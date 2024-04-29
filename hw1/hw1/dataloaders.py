import math
import numpy as np
import torch
import torch.utils.data
from typing import Sized, Iterator
from torch.utils.data import Dataset, Sampler, SubsetRandomSampler, DataLoader



class FirstLastSampler(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        # TODO:
        # Implement the logic required for this sampler.
        # If the length of the data source is N, you should return indices in a
        # first-last ordering, i.e. [0, N-1, 1, N-2, ...].
        # ====== YOUR CODE: ======
<<<<<<< HEAD
        n = len(self.data_source)
        return (i if i % 2 == 0 else n - 1 - i // 2 for i in range(n))
=======

        n = len(self.data_source)
        # Yielding indices from start to end and from end to start converging towards the center
        start = 0
        end = n - 1
        while start <= end:
            yield start
            if start != end:
                yield end
            start += 1
            end -= 1
>>>>>>> LNN-3600/master
        # ========================
    def __len__(self):
        return len(self.data_source)


def create_train_validation_loaders(
    dataset: Dataset, validation_ratio, batch_size=100, num_workers=2
):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # TODO:
    #  Create two DataLoader instances, dl_train and dl_valid.
    #  They should together represent a train/validation split of the given
    #  dataset. Make sure that:
    #  1. Validation set size is validation_ratio * total number of samples.
    #  2. No sample is in both datasets. You can select samples at random
    #     from the dataset.
    #  Hint: you can specify a Sampler class for the `DataLoader` instance
    #  you create.
    # ====== YOUR CODE: ======
<<<<<<< HEAD
    n_total = len(dataset)
    n_valid = int(n_total * validation_ratio)
    n_train = n_total - n_valid
    indices = list(range(n_total))
    indices_tensor = torch.tensor(indices)
    indices = indices_tensor[torch.randperm(len(indices_tensor))]
    train_indices = indices[:n_train]
    valid_indices = indices[n_train:]


    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_indices)

    dl_train = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler
    )
    dl_valid = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, sampler=valid_sampler
    )
=======

    # Total number of samples in the dataset
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split = int(np.floor(validation_ratio * num_samples))

    # Shuffle indices randomly
    np.random.shuffle(indices)

    # Split indices into training and validation sets
    train_indices, val_indices = indices[split:], indices[:split]

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    # Create DataLoader instances
    dl_train = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    dl_valid = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler, num_workers=num_workers)

>>>>>>> LNN-3600/master
    # ========================

    return dl_train, dl_valid
