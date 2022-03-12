import torch
import numpy as np
import matplotlib.pyplot as plt
import math


def is_prime(n: int):
    if n == 1:
        return False, 1
    elif n == 4:
        return False, 2
    for ii in range(2, math.ceil(math.sqrt(n))):
        if n % ii == 0:
            return False, ii
    return True, n


def imshow(inputs, nrow=None, ncol=None):
    """
    Show the images.

    Inputs
    ------
    inputs: torch.tensor or numpy.array
    """
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().numpy()

    if (inputs.ndim > 3 or inputs.ndim < 2):
        raise ValueError("Input must be 3 dimensional or 2 dimensional but, "
                        f"receive input of shape{inputs.shape}""")
    elif inputs.ndim == 2:
        plt.imshow(inputs)
        plt.show()
    elif inputs.ndim == 3:
        if nrow:
            if ncol:
                assert (inputs.shape[0] == nrow * ncol), \
                    "nrow * ncol must equal inputs.shape[0]"
            assert (inputs.shape[0] % nrow == 0), \
                "Value of nrow must divide inputs.shape[0]"
        else:
            if is_prime(inputs.shape[0])[0]:
                # assert (inputs.shape[0] < 7), \
                # "inputs.shape[0] is a prime number, please specify nrow (and ncol)"
                print("inputs.shape[0] is a prime number")
                nrow = inputs.shape[0]
            else: 
                nrow = is_prime(inputs.shape[0])[1]
        ncol = inputs.shape[0] // nrow
        fig, ax = plt.subplots(nrow, ncol)
        print(f"Showing {inputs.shape[0]} 2d map in {nrow} x {ncol} grid.")
        if ncol != 1:
            for ii in range(inputs.shape[0]):
                ax[ii // ncol, ii % ncol].imshow(inputs[ii])
        else:
            for ii in range(inputs.shape[0]):
                ax[ii].imshow(inputs[ii])
        plt.show()


if __name__ == "__main__":
    a = torch.randn(12, 32, 32)
    imshow(a, nrow=3)
