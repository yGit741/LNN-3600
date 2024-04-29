import torch
from torch import Tensor
from typing import Tuple, Iterator
from contextlib import contextmanager
from torch.utils.data import Dataset, IterableDataset


def random_labelled_image(
    shape: Tuple[int, ...], num_classes: int, low=0, high=255, dtype=torch.int,
) -> Tuple[Tensor, int]:
    """
    Generates a random image and a random class label for it.
    :param shape: The shape of the generated image e.g. (C, H, W).
    :param num_classes: Number of classes. The label should be in [0, num_classes-1].
    :param low: Minimal value in the image (inclusive).
    :param high: Maximal value in the image (exclusive).
    :param dtype: Data type of the returned image tensor.
    :return: A tuple containing the generated image tensor and it's label.
    """
    # TODO:
    #  Implement according to the docstring description.
    # ====== YOUR CODE: ======
<<<<<<< HEAD
    image = torch.randint(low, high, shape, dtype=dtype)
    label = torch.randint(0, num_classes, (1,)).item()
=======

    # Generate the image tensor with random values between low and high
    image = torch.randint(low=low, high=high, size=shape, dtype=dtype)

    # Generate a random label between 0 and num_classes-1
    label = torch.randint(0, num_classes, (1,)).item()

>>>>>>> LNN-3600/master
    # ========================
    return image, label

@contextmanager
def torch_temporary_seed(seed: int):
    """
    A context manager which temporarily sets torch's random seed, then sets the random
    number generator state back to its previous state.
    :param seed: The temporary seed to set.
    """
    # TODO:
    #  Implement this context manager as described.
    #  See torch.random.get/set_rng_state(), torch.random.manual_seed().
    # ====== YOUR CODE: ======
<<<<<<< HEAD
    state = torch.random.get_rng_state()
    # ========================
    try:
        # ====== YOUR CODE: ======
        torch.random.manual_seed(seed)
=======
    # Save the current state of PyTorch's random number generator
    orig_state = torch.get_rng_state()

    # ========================
    try:
        # ====== YOUR CODE: ======
        # Set the new random seed
        torch.manual_seed(seed)
>>>>>>> LNN-3600/master
        # ========================
        yield
    finally:
        # ====== YOUR CODE: ======
<<<<<<< HEAD
        torch.random.set_rng_state(state)
        # ========================
=======
        # Restore the original state of the random number generator
        torch.set_rng_state(orig_state)
    # ========================
>>>>>>> LNN-3600/master


class RandomImageDataset(Dataset):
    """
    A dataset representing a set of noise images of specified dimensions.
    """

    def __init__(self, num_samples: int, num_classes: int, C: int, W: int, H: int):
        """
        :param num_samples: Number of samples (labeled images in the dataset)
        :param num_classes: Number of classes (labels)
        :param C: Number of channels per image
        :param W: Image width
        :param H: Image height
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.image_dim = (C, W, H)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        """
        Returns a labeled sample.
        :param index: Sample index.
        :return: A tuple (sample, label) containing the image and its class label.
        Raises a ValueError if index is out of range.
        """

        # TODO:
        #  Create a random image tensor and return it.
        #  Make sure to always return the same image for the
        #  same index (make it deterministic per index), but don't mess-up
        #  the random state outside this method.
        #  Raise a ValueError if the index is out of range.
        # ====== YOUR CODE: ======
<<<<<<< HEAD

        # verifying the index and raise 'Index out of range' error if necessary
        if index >= self.num_samples or index < 0:
            raise ValueError('Index out of range')

        # set an index based deterministic seed
        with torch_temporary_seed(index):
            # returning the labeled image
            return random_labelled_image(shape=self.image_dim, num_classes=self.num_classes)
=======
        if index >= self.num_samples:
            raise ValueError('Index is out of range')
        else:
            with torch_temporary_seed(index):
                return random_labelled_image(self.image_dim, self.num_classes)
>>>>>>> LNN-3600/master
        # ========================

    def __len__(self):
        """
        :return: Number of samples in this dataset.
        """
        # ====== YOUR CODE: ======
        return self.num_samples
        # ========================


class ImageStreamDataset(IterableDataset):
    """
    A dataset representing an infinite stream of noise images of specified dimensions.
    """

    def __init__(self, num_classes: int, C: int, W: int, H: int):
        """
        :param num_classes: Number of classes (labels)
        :param C: Number of channels per image
        :param W: Image width
        :param H: Image height
        """
        super().__init__()
        self.num_classes = num_classes
        self.image_dim = (C, W, H)

    def __iter__(self) -> Iterator[Tuple[Tensor, int]]:
        """
        :return: An iterator providing an infinite stream of random labelled images.
        """

        # TODO:
        #  Yield tuples to produce an iterator over random images and labels.
        #  The iterator should produce an infinite stream of data.
        # ====== YOUR CODE: ======
        while True:
<<<<<<< HEAD
            # generate a random index to use with torch_temporary_seed
            index = torch.randint(0, self.num_classes, (1,)).item()

            # use torch_temporary_seed  to ensure reproducibility
            with torch_temporary_seed(index):
                # generate a random labeled image using the given index
                image, label = random_labelled_image(shape=self.image_dim, num_classes=self.num_classes)

            # yield the image and label as a tuple
            yield image, label
=======
            image, label = random_labelled_image(self.image_dim, self.num_classes)
            yield (image, label)
>>>>>>> LNN-3600/master
        # ========================


class SubsetDataset(Dataset):
    """
    A dataset that wraps another dataset, returning a subset from it.
    """

    def __init__(self, source_dataset: Dataset, subset_len: int, offset=0):
        """
        Create a SubsetDataset from another dataset.
        :param source_dataset: The dataset to take samples from.
        :param subset_len: The total number of sample in the subset.
        :param offset: The offset index to start taking samples from.
        """
        if offset + subset_len > len(source_dataset):
            raise ValueError("Not enough samples in source dataset")

        self.source_dataset = source_dataset
        self.subset_len = subset_len
        self.offset = offset

    def __getitem__(self, index):
        # TODO:
        #  Return the item at index + offset from the source dataset.
        #  Raise an IndexError if index is out of bounds.
        # ====== YOUR CODE: ======
<<<<<<< HEAD
        # check if the index is the range of the subset
        if index >= self.subset_len:
            raise IndexError("Index out of bounds")
        # return the item at index + offset from  source dataset
        return self.source_dataset[index + self.offset]
=======
        if index + self.offset >= self.subset_len:
            raise IndexError("Index out of range")
        else:
            return self.source_dataset[self.offset + index]
>>>>>>> LNN-3600/master
        # ========================

    def __len__(self):
        # ====== YOUR CODE: ======
        return self.subset_len
        # ========================
