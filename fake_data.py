import numpy as np
import matplotlib.pylab as plt


def model(t_vec, lambda_vec, A_vec):
    '''
    vectorized implementation to create model observation data.

    inputs:
    t_vec: [k,1] vector of observation times
    lambda_vec: [m, 1] vector of lambda parameters
    A_vec: [m, 1] vector of A parameters

    returns:
    data: [k,1] vector of modeled observation points of form
        data[k,1] = sum(A_i * e^{-lambda_i * t_k}) for i in 1,...,k
    '''

    data = np.matmul(
        np.exp(
            np.matmul(
                t_vec,
                -lambda_vec.transpose()
            )
        ),
        A_vec
    )

    return data


def generate_data(t_vec, lambda_vec, A_vec, sigma):
    '''
    Add gaussian noise to model data

    inputs:
    t_vec: [k,1] vector of observation times
    lambda_vec: [m, 1] vector of lambda parameters
    A_vec: [m, 1] vector of A parameters
    sigma: scalar value for standard deviation of added noise

    returns:
    Y_vec: [k,1] vector of noisy observations
    '''
    data = model(t_vec, lambda_vec, A_vec)
    noise = np.random.normal(0, sigma, data.shape)
    Y_vec = data + noise
    return(Y_vec)


def plot(t_vec, lambda_vec, A_vec, sigma):
    '''
    function to plot noisy observations and de-noised model curve
    '''

    t_fine = np.linspace(
        min(t_vec),
        max(t_vec),
        100
        ).reshape(100, 1)
    model_points = model(t_fine, lambda_vec, A_vec)
    data_points = generate_data(t_vec, lambda_vec, A_vec, sigma)
    plt.plot(t_fine, model_points, label='model')
    plt.scatter(t_vec, data_points, label='noisy data', color='r')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.savefig('model_hard.pdf')
    plt.show()
