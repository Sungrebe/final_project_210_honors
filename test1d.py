import numpy as np
from filter import *
from matplotlib import pyplot as plt

# This is for you to make sure your code works 
# Feel free to modify this however you wish :)
def main():
    # 1D Test
    x = np.zeros(1000)
    x[:1000] = np.arange(1000) # create test signal
    sigma = 250

    w = linear_correlation1D(x, sigma, 0)

    i = 50
    z = np.roll(x, i) # test detection space

    detected = detect1D(w, z)
    print(f"detected x at {detected}, actual position: {i}")

    # 2D Test

    data = np.arange(625).reshape(25, 25)
    sigma = 10

    plt.matshow(create_gaussian_target(data, sigma))
    plt.colorbar()
    plt.show()

    """
    array_1 = np.zeros(1000)
    array_1[:1000] = np.arange(1000)

    print(array_1)
    print(create_gaussian_target1D(array_1, sigma))

    data = np.arange(625).reshape(25, 25)

    plt.matshow(create_gaussian_target(data, sigma))
    plt.colorbar()
    plt.show()
    """

if __name__ == '__main__':
    main()