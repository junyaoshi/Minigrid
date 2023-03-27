import numpy as np

def generate_feedback(value, noise=0.05, hd_feedback="None"):
    if hd_feedback == "base2":
        return noisy_base2_feedback(value, noise)
    elif hd_feedback == "mnist":
        return 0
    else:
        return noisy_sigmoid_feedback(value, noise)


def noisy_sigmoid_feedback(value, noise=0.05):
    if isinstance(value, np.ndarray):  # vectorized
        return 1 / (1 + np.exp(-value)) + np.random.normal(0, noise, value.shape)
    else:  # scalar
        return 1 / (1 + np.exp(-value)) + np.random.normal(0, noise)
    

def noisy_base2_feedback(value, noise=0.05, n_bits=16):
    value_with_noise = np.random.normal(value, noise)
    # print(value_with_noise.shape)
    if isinstance(value_with_noise, np.ndarray):
        vectorized_binary = np.vectorize(create_binary, excluded=['n_bits'], signature='()->(n)')
        # Apply the vectorized function to the ndarray
        result = vectorized_binary(value_with_noise, n_bits=n_bits)
        # Reshape the result to have the same number of dimensions as the original array, plus one for the size-2 arrays
        feedback = result.reshape(*value_with_noise.shape, n_bits)
        # print(feedback.shape)
        return feedback
    else:
        feedback = create_binary(value_with_noise, n_bits)
        return feedback

def noisy_mnist_feedback(value, noise=0.05):
    # convert feedback to mnist image
    pass

def create_binary(x, n_bits=16):
    if n_bits == 16:
        binary = bin(np.float16(x).view('H'))[2:].zfill(16)
    elif n_bits == 32:
        binary = bin(np.float32(x).view('I'))[2:].zfill(32)
    else:
        raise ValueError("n_bits must be 16 or 32")
    return np.array([int(bit) for bit in binary])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_sigmoid = False
    test_base2 = True

    # Use 25 1s and 25 -1s, plot the noisy sigmoid feedback
    if test_sigmoid:
        vals = np.ones(1000)
        vals[:500] *= -1
        feedback = noisy_sigmoid_feedback(vals)
        plt.scatter(vals, feedback, s=2)
        plt.savefig("feedback.png")
    
    # Convert a float to a 16 bit binary string
    if test_base2:
        vals = 5
        feedback = noisy_base2_feedback(vals)
        print(feedback)

