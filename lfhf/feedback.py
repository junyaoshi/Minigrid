import numpy as np

def noisy_sigmoid_feedback(value):
    return 1/(1 + np.exp(-value)) + np.random.normal(0, 0.05)


if __name__ == "__main__":
    # Use 25 1s and 25 -1s, plot the noisy sigmoid feedback
    import matplotlib.pyplot as plt
    vals = np.ones(50)
    vals[:25] *= -1
    feedback = []
    for val in vals:
        feedback.append(noisy_sigmoid_feedback(val))
    plt.scatter(vals, feedback, s=2)
    plt.savefig('feedback.png')
