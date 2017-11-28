from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import math


def calc_rho(x, m, ts):

    t_max = len(ts)
    C_0 = np.square(np.std(x, axis=1)).reshape(m, 1)
    C_t = np.zeros((m, t_max))  # [m,t_max]

    for t in ts[:-1]:
        C_t[:, t] = (1/(t_max-t))*np.sum(
            np.multiply(
                (x[:, 0:t_max-t] - np.mean(x[:, 0:t_max-t], axis=1).reshape(m, 1)),
                (x[:, t:] - np.mean(x[:, 0:t_max-t], axis=1).reshape(m, 1))
                ),
            axis=1
            )


    rho_t = np.divide(C_t, C_0)

    return rho_t


def tau_self_consistent(rho_t, w=3):
    m = rho_t.shape[0]
    tau_sc = np.ones((m, 1))

    for i in range(m):
        t = 0
        count = 0
        while w*tau_sc[i, 0] > t:
            count +=1
            t += 1
            tau_sc[i, 0] += rho_t[i, t]
        print("tau count %s" %count)
    return tau_sc


def plot_roes(lambda_walk, A_walk, m, t_max, burnin):

    lambda_walk = lambda_walk[:,burnin:t_max]
    A_walk = A_walk[:,burnin:t_max]
    ts = [t for t in range(0, t_max-burnin)]

    lambda_roes = calc_rho(lambda_walk, m, ts)
    A_roes = calc_rho(A_walk, m, ts)
    tau_lambda_sc = tau_self_consistent(lambda_roes)
    tau_A_sc = tau_self_consistent(A_roes)

    tau_lambda_ac = 1 + 2*np.sum(lambda_roes,axis=1)
    tau_A_ac = 1 + 2*np.sum(A_roes,axis=1)

    print("tau_lambda_ac: %s" %tau_lambda_ac)
    print("tau_lambda_sc: %s" %tau_lambda_sc)
    print("tau_A_ac: %s" %tau_A_ac)
    print("tau_A_sc: %s" %tau_A_sc)

    fig, axes = plt.subplots(2, m)
    for i, ax in enumerate(axes.flatten()):
        if i < m:
            ax.plot(ts, lambda_roes[i])
            ax.set_title("lambda_%s\ntau_sc =%i" %(i,int(tau_lambda_sc[i])))
        else:
            ax.plot(ts, A_roes[i-m])
            ax.set_title("A_%s\ntau_sc=%i" % ((i-m),int(tau_A_sc[i-m])))
    fig.suptitle(
        "rho_t for posterior samples"
        )
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig('rho_easy.pdf')
    plt.show()

    return (tau_lambda_sc, tau_A_sc)
