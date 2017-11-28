import matplotlib.pylab as plt
import numpy as np
import fake_data as fd

def draw_walks(m, lambda_values, A_values):
    '''
    plot the random walk for each model parameter
    '''
    fig, axes = plt.subplots(2, m)
    for i, ax in enumerate(axes.flatten()):
        if i < m:
            ax.plot(lambda_values[i, :])
            ax.set_title("lambda_%s" % i)
        else:
            ax.plot(A_values[i-m, :])
            ax.set_title("A_%s" % (i-m))
    fig.suptitle(
        "MCMC random walk for model parameter values (N = %s)"
        % lambda_values.shape[1]
        )
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig('walks_easy.pdf')
    plt.show()


def draw_hists(
    m,
    lambda_values,
    A_values,
    lambda_tau,
    A_tau,
    t_vec,
    y_vec,
    lambda_exact,
    A_exact,
    sigma_exact,
    burnin=50000
):

    '''
    Produces histograms of model parameters, sampling posterior
    distributions once every ac_step

    Displays residuals for original observation data for exact model
    max-liklihood parameter values, and mean parameter values
    '''
    k = lambda_values.shape[1]
    fig, axes = plt.subplots(2, m)
    lambda_max = np.zeros((m, 1))
    A_max = np.zeros((m, 1))
    for i, ax in enumerate(axes.flatten()):
        if i < m:
            samples = lambda_values[i, np.arange(
                burnin, k, int(lambda_tau[i])
                )]
            n_eff = len(samples)
            n, bins, _ = ax.hist(samples)
            ax.set_title("lambda_%s\nN_eff=%i" % (i,int(n_eff)))
            lambda_max[i] = sorted(
                zip(n, bins),
                key=lambda x: x[0],
                reverse=True
                )[0][1] + (bins[1]-bins[0])/2
        else:
            samples = A_values[i-m, np.arange(
                burnin, k, int(A_tau[i-m])
                )]
            n_eff = len(samples)
            n, bins, _ = ax.hist(samples)
            A_max[i-m] = sorted(
                zip(n, bins),
                key=lambda x: x[0],
                reverse=True
                )[0][1] + (bins[1]-bins[0])/2
            ax.set_title("A_%s\nN_eff=%i" % ((i-m),int(n_eff)))

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig('hists_easy.pdf')
    plt.show()

    t_fine = np.linspace(
        min(t_vec),
        max(t_vec),
        100
        ).reshape(100,1)

    model_max_points = fd.model(t_fine, lambda_max, A_max)
    exact_points = fd.model(t_fine, lambda_exact, A_exact)

    y_exact = fd.model(t_vec, lambda_exact, A_exact)
    y_max = fd.model(t_vec, lambda_max, A_max)
    e_exact = y_exact - y_vec
    e_max = y_max - y_vec
    print("L2 ERROR FOR MODEL %4f" % np.linalg.norm(e_exact))
    print("L2 ERROR FOR MAX %4f" % np.linalg.norm(e_max))

    plt.plot(t_fine, model_max_points, label='max_L (e = %.2f)' %np.linalg.norm(e_max))
    plt.plot(t_fine, exact_points, 'r', label='model (e = %.2f)' %np.linalg.norm(e_exact))
    plt.scatter(t_vec, y_vec)
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,-2,10))
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig('resid_easy.pdf')
    plt.show()
    return (lambda_max, A_max)
