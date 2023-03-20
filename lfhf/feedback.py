import numpy as np


def noisy_sigmoid_feedback(value, noise=0.05):
    if isinstance(value, np.ndarray):  # vectorized
        return 1 / (1 + np.exp(-value)) + np.random.normal(0, noise, value.shape)
    else:  # scalar
        return 1 / (1 + np.exp(-value)) + np.random.normal(0, noise)
    

def noisy_base2_feedback(value, noise=0.05):
    # convert base 10 to base 2
    pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_sigmoid = True

    # Use 25 1s and 25 -1s, plot the noisy sigmoid feedback
    if test_sigmoid:
        vals = np.ones(1000)
        vals[:500] *= -1
        feedback = noisy_sigmoid_feedback(vals)
        plt.scatter(vals, feedback, s=2)
        plt.savefig("feedback.png")
