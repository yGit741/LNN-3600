import torch



class TensorView(object):
    """
    A transform that returns a new view of a tensor.
    """

    def __init__(self, *view_dims):
        self.view_dims = view_dims

    def __call__(self, tensor: torch.Tensor):  
        return tensor.view(*self.view_dims)

class InvertColors(object):
    """
    Inverts colors in an image given as a tensor.
    """

    def __call__(self, x: torch.Tensor):
        """
        :param x: A tensor of shape (C,H,W) for values in the range [0, 1],
            representing an image.
        :return: The image with inverted colors.
        """
        # TODO: Invert the colors of the input image.
        # ====== YOUR CODE: ======
<<<<<<< HEAD
        # Invert the color due to the 0 to 1 normalization
=======

        # Check if the input tensor is a floating point data type
        if not torch.is_floating_point(x):
            raise TypeError("Input tensor must be a floating point tensor")

        # Ensure the values are within the expected range [0, 1]
        if x.min() < 0 or x.max() > 1:
            raise ValueError("Input tensor values should be in the range [0, 1]")

        # Invert the colors
>>>>>>> LNN-3600/master
        return 1 - x
        # ========================


class FlipUpDown(object):
    def __call__(self, x: torch.Tensor):
        """
        :param x: A tensor of shape (C,H,W) representing an image.
        :return: The image, flipped around the horizontal axis.
        """
        # TODO: Flip the input image so that up is down.
        # ====== YOUR CODE: ======
<<<<<<< HEAD
        # use the builtin torch function
        return torch.flip(x, [1])
=======

        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor")
        if x.dim() != 3:
            raise ValueError("Input tensor must have three dimensions (C, H, W)")

            # Flip the image vertically
        return x.flip(1)  # Flips along the height dimension
>>>>>>> LNN-3600/master
        # ========================


class BiasTrick(object):
    """
    A transform that applies the "bias trick": Prepends an element equal to
    1 to each sample in a given tensor.
    """

    def __call__(self, x: torch.Tensor):
        """
        :param x: A pytorch tensor of shape (D,) or (N1,...Nk, D).
        We assume D is the number of features and the N's are extra
        dimensions. E.g. shape (N,D) for N samples of D features;
        shape (D,) or (1, D) for one sample of D features.
        :return: A tensor with D+1 features, where a '1' was prepended to
        each sample's feature dimension.
        """
        assert x.dim() > 0, "Scalars not supported"

        # TODO:
        #  Add a 1 at the beginning of the given tensor's feature dimension.
        #  Hint: See torch.cat().
        # ====== YOUR CODE: ======
<<<<<<< HEAD
        # add a new dimension for the bias term at index 1 (after the batch dimension if it exists)
        ones = torch.ones(*x.shape[:-1], 1, dtype=x.dtype)

        # concatenate the bias term with the original tensor along the last dimension
        return torch.cat((ones, x), dim=-1)
        # ========================
=======
        # Get the size of all dimensions except for the last one
        shape_of_ones = x.shape[:-1] + (1,)  # creates a new tuple with a 1 in the last position

        # Create a tensor of ones that matches the shape for the prepend operation
        ones = torch.ones(shape_of_ones, dtype=x.dtype, device=x.device)

        # Concatenate the ones tensor and the input tensor along the last dimension
        result = torch.cat([ones, x], dim=-1)

    # ========================
        return result
>>>>>>> LNN-3600/master
