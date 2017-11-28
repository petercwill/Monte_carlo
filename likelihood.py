from __future__ import division
import fake_data
import math
import numpy as np


def gauss_pdf(x_vec, sigma):
    '''
    Evaluates p(x) where x represents model residual for data point.

    input:
      x_vec: [k,1] vector of residuals
      sigma: standard deviation of gaussian

    output:
      prob_vec: [k,1] vector of probabilites for residuals.
    '''
    prob_vec = (1 / (2*math.pi*sigma**2)**.5)*np.exp(
        -.5*np.square(x_vec)/(2*sigma**2)
        )

    return prob_vec


def b_prior(lambda_vec, A_vec, sigma):
    '''
    prior distribution for model parameters.
    Represent uninformative prior, with sigma
    restricted to positive values
    '''
    return 1
    # if np.all(lambda_vec > 0) and np.all(A_vec > 0) and sigma > 0:
    #     return 1
    # else:
    #     return 0


def L(t_vec, Y_vec, lambda_vec, A_vec, sigma):
    '''
    Evaluates likelihood function for proposed model parameters.

    input:
      t_vec: [k,1] vector of observation times
      Y_vec: [k,1] vector of observation data
      lambda_vec: [m,1] vector of lambda model parameters
      A_vec: [m,1] vector of A model parameters
      sigma: scalar value for sigma modlel parameter

    output:
        returns a scalar probability of proposed model parameters
        given observed data.  L(lambda_vec, A_vec, sigma | Y_vec)
    '''

    noise = Y_vec - fake_data.model(t_vec, lambda_vec, A_vec)
    prob_vec = gauss_pdf(noise, sigma)
    prior = b_prior(lambda_vec, A_vec, sigma)

    return prior*np.prod(prob_vec)
