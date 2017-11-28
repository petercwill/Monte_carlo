from __future__ import division
import numpy as np
import likelihood as li
import random as rn
import fake_data as fd
import matplotlib.pylab as plt
import graphs as gr
import auto_correlation as ac

def iso_gauss(lambda_vec, A_vec, sigma, r):
    '''
    proposal distribution for parameters.  isotropic gaussian
    with step size of r.

    input:
      lambda_vec: [m,1] vector of current lambda parameter values
      A_vec: [m,1] vector of current A proposed values
      sigma: scalar current sigma value
      r: scalar value for magnitude of covariance matrix

    output:
      lambda_vec_star: proposed sample for lambda_vec
      A_vec_star: proposed sample for A_vec
      sigma_star: proposed sample for sigma
      prob_1: proposal probability P(x_(i-1) | x_i)
      prob_2: proposal probability P(x_i | x_(i-1))

    '''
    sigma_vec = np.array([[sigma]])
    n = lambda_vec.shape[0] + A_vec.shape[0] + sigma_vec.shape[0]
    mu = np.concatenate((lambda_vec, A_vec, sigma_vec)).reshape(n)
    cov = r*np.identity(n)
    #cov = np.diag([.01,.01,.01,.1,.1,.1,.1])
    y = np.random.multivariate_normal(mu, cov).reshape(n, 1)
    lambda_vec_star = y[:lambda_vec.shape[0]]
    A_vec_star = y[lambda_vec.shape[0]:-1]
    sigma_star = y[-1][0]

    return (
        lambda_vec_star,
        A_vec_star,
        sigma_star,
        )


def MRT(target, proposal, lambda_vec, A_vec, sigma, t_vec, Y_vec):
    '''
    Metropolis Hasting Algorithm

    input:
      target: likelihood function of form
        L(t_vec, Y_vec, lambda_vec, A_vec, sigma)
        evaluating probability of parameters for observational data

      proposal: proposal distribution of form
        Q(lambda_vec, A_vec, sigma, r)

      lambda_vec: [m,1] vector of current lambda parameter values
      A_vec: [m,1] vector of current A parameter values
      sigma: scalar of current sigma value
      t_vec: [k,1] vector of observational time values
      Y_vec: [k,1] vector of observational data values

    output:
      tuple of parameter values for next MCMC step.
    '''
    lambda_prop, A_prop, sigma_prop = proposal(
        lambda_vec,
        A_vec,
        sigma,
        .1
        )

    q1 = target(
        t_vec,
        Y_vec,
        lambda_vec,
        A_vec,
        sigma
        )
    q2 = target(
        t_vec,
        Y_vec,
        lambda_prop,
        A_prop,
        sigma_prop
        )
    accept_prop = min(1, q2/q1)
    # if q2==0 and q1==0:
    #     accept_prop = 0
    accept_flag = False

    if rn.uniform(0, 1) <= accept_prop:
        accept_flag = True
        return (lambda_prop, A_prop, sigma_prop, accept_flag)

    else:
        return(lambda_vec, A_vec, sigma, accept_flag)


def mcmc(
    walk_length,
    y_vec,
    t_vec,
    m,
    lambda_vec0,
    A_vec0,
    sigma0
    ):

    target = li.L
    proposal = iso_gauss
    sigma = sigma0
    lambda_vec = lambda_vec0
    A_vec = A_vec0

    lambda_walk = np.zeros((m, walk_length))
    A_walk = np.zeros((m, walk_length))
    sigma_walk = np.zeros(walk_length)
    accept_counter = 0
    for i in range(walk_length):

        if i % (walk_length/10) == 0:
            print("step: %s" % i)

        lambda_vec, A_vec, sigma, accept_flag = MRT(
            target,
            proposal,
            lambda_vec,
            A_vec,
            sigma,
            t_vec,
            y_vec
            )

        if accept_flag:
            accept_counter += 1

        lambda_walk[:, i, np.newaxis] = lambda_vec
        A_walk[:, i, np.newaxis] = A_vec
        sigma_walk[i] = sigma

    print("accepted %.2f percent of the time" % (
        round(accept_counter/walk_length, 4)*100
        ))
    return lambda_walk, A_walk, sigma_walk


def main():
    problem = 'easy'

    # vector of observation times
    t_vec = np.linspace(0, 10, 11).reshape(11, 1)

    # problem specific parameters
    if problem == 'hard':
        lambda_vec = np.array([[.1], [.3], [.8]])
        A_vec = np.array([[4], [-5], [6]])
        m = lambda_vec.shape[0]
        sigma = .2
    if problem == 'easy':
        lambda_vec = np.array([[.25]])
        A_vec = np.array([[7]])
        m = lambda_vec.shape[0]
        sigma = 1

    fd.plot(t_vec, lambda_vec, A_vec, sigma)
    # MCMC settings
    walk_length = 40000
    burnin = int(.2*walk_length)  # burn-in, discard the first 20% of samples

    # initial guesses
    lambda_0 = np.ones((m, 1))
    A_0 = np.ones((m, 1))
    sigma_0 = 1

    Y_vec = fd.generate_data(t_vec, lambda_vec, A_vec, sigma)
    lambda_walk, A_walk, sigma_walk = mcmc(
        walk_length,
        Y_vec,
        t_vec,
        m,
        lambda_0,
        A_0,
        sigma_0
        )

    gr.draw_walks(m, lambda_walk, A_walk)

    tau_lambda, tau_A = ac.plot_roes(
        lambda_walk,
        A_walk,
        m,
        t_max=walk_length,
        burnin=burnin
        )

    # auto_correlation_length = int(
    #     np.max(
    #         np.concatenate((tau_lambda, tau_A))
    #         ))
    #
    # print("tau = %s" % auto_correlation_length)
    # print(
    #     "effective sample size = %i"
    #     % int(walk_length/auto_correlation_length)
    # )

    lambda_star, A_star = gr.draw_hists(
        m,
        lambda_walk,
        A_walk,
        tau_lambda,
        tau_A,
        t_vec,
        Y_vec,
        lambda_vec,
        A_vec,
        sigma,
        burnin
        )

    print("Lambda_Star: %s" %lambda_star)
    print("A_Star: %s" %A_star)

if __name__ == '__main__':
    main()
