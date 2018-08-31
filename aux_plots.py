from scipy.stats import expon
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn') # pretty matplotlib plots
plt.rcParams['figure.figsize'] = (12, 8)
import scipy.stats as ss
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


def plot_log_likelihood(df):
    th_0, th_1 = np.arange(0, 3, 0.05), np.arange(0, 3, 0.05)
    Z = np.array([-sum((df.y- t0 - t1*df.x)**2) 
                  for t0 in th_0 for t1 in  th_1 ]).reshape((len(th_0),len(th_1)))
    t0_best, t1_best = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
    t0_best, t1_best = th_0[t0_best], th_1[t1_best]
    fig = plt.figure(figsize=(15,8))
    ax = fig.gca(projection='3d')
    ax.contour3D(th_0, th_1, Z, 200, cmap='coolwarm')
    ax.scatter(t1_best, t0_best , Z.max(), c='g', marker='x')
    ax.set_xlabel(r'$\theta_1$'), ax.set_ylabel(r'$\theta_0$'), ax.set_zlabel(r'$L(D|\theta)$'), ax.view_init(elev=45., azim=11.0)
    #print(f"th0= {t0_best}, th1= {t1_best}, log_max = {Z.max()} ")
    plt.show()
    return t0_best, t1_best
    
    
    

def simulate_linear_data(N, beta_0, beta_1, eps_sigma_sq):
    """
    Simulate a random dataset using a noisy
    linear process.
    N: Number of data points to simulate
    theta_0: Intercept
    theta_1: Slope of univariate predictor, X
    """
    # Create a pandas DataFrame with column 'x' containing
    # N uniformly sampled values between 0.0 and 1.0
    df = pd.DataFrame(
        {"x": 
            np.random.RandomState(42).choice(np.arange(0, 1, 1/N), N, replace=False
            )
        }
    )

    # Use a linear model (y ~ beta_0 + beta_1*x + epsilon) to 
    # generate a column 'y' of responses based on 'x'
    eps_mean = 0.0
    err = np.random.RandomState(42).normal(eps_mean, eps_sigma_sq, N)
    df["y"] = beta_0 + beta_1*df["x"] + err
    plt.figure(figsize=(15, 8));
    plt.scatter(df.x,df.y);
    plt.plot(df.x,beta_0 + beta_1*df.x, c='red', label="True Regression Line");
    plt.legend(prop={'size': 12});
    return df, err

def plot_discrete(xk, pk):
    '''
    Plots the exponential distribution function for a given x range
    If mu and sigma are not provided, standard exponential is plotted
    If cdf=True cumulative distribution is plotted
    Passes any keyword arguments to matplotlib plot function
    '''
    custm = ss.rv_discrete(name='custm', values=(xk, pk))
    plt.plot(xk, custm.pmf(xk), 'go', ms=8, alpha=0.5,mec='green', label='Discrete')
    plt.vlines(xk, 0, custm.pmf(xk), colors='green', alpha=0.5,lw=2)
    plt.legend( loc=1,);


def plot_exponential(x_range, mu=0, sigma=1, cdf=False, **kwargs):
    '''
    Plots the exponential distribution function for a given x range
    If mu and sigma are not provided, standard exponential is plotted
    If cdf=True cumulative distribution is plotted
    Passes any keyword arguments to matplotlib plot function
    '''
    x = x_range
    if cdf:
        y = ss.expon.cdf(x, mu, sigma)
    else:
        y = ss.expon.pdf(x, mu, sigma)
    plt.plot(x, y, **kwargs)
    plt.legend();
    

def plot_beta(x_range, a, b, mu=0, sigma=1, cdf=False, **kwargs):
    '''
    Plots the f distribution function for a given x range, a and b
    If mu and sigma are not provided, standard beta is plotted
    If cdf=True cumulative distribution is plotted
    Passes any keyword arguments to matplotlib plot function
    '''
    x = x_range
    if cdf:
        y = ss.beta.cdf(x, a, b, mu, sigma)
    else:
        y = ss.beta.pdf(x, a, b, mu, sigma)
    plt.plot(x, y, **kwargs)
    plt.legend();
    
def plot_normal(x_range, mu=0, sigma=1, cdf=False, **kwargs):
    '''
    Plots the normal distribution function for a given x range
    If mu and sigma are not provided, standard normal is plotted
    If cdf=True cumulative distribution is plotted
    Passes any keyword arguments to matplotlib plot function
    '''
    x = x_range
    if cdf:
        y = ss.norm.cdf(x, mu, sigma)
    else:
        y = ss.norm.pdf(x, mu, sigma)
    plt.plot(x, y, **kwargs)
    plt.legend();