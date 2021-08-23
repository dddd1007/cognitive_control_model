from __future__ import division
import numpy as np
from numpy import log, exp, power, pi
from numpy.lib.function_base import select
from scipy.special import gammaln
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    import warnings
    message = "Could not import seaborn; plotting will not work."
    warnings.warn(message, UserWarning)

# Helper Functions -----------------------------------------------------------

def make_grid(start, stop, step):
    """Define an even grid over a parameter space."""
    count = (stop - start) / step + 1
    return np.linspace(start, stop, int(count))


def I_trans_func(I_p1, I, k):
    """I_p1 is normal with mean I and std dev k."""
    var = exp(k * 2)
    pdf = exp(-.5 * power(I - I_p1, 2) / var)
    pdf *= power(2 * pi * var, -0.5)
    return pdf


def p_trans_func(p_p1, p, I_p1):
    """p_p1 is beta with mean p and precision I_p1."""
    a = 1 + exp(I_p1) * p
    b = 1 + exp(I_p1) * (1 - p)

    if 0 < p_p1 < 1:
        logkerna = (a - 1) * log(p_p1)
        logkernb = (b - 1) * log(1 - p_p1)
        betaln_ab = gammaln(a) + gammaln(b) - gammaln(a + b)
        return exp(logkerna + logkernb - betaln_ab)
    else:
        return 0

# Model Functions -----------------------------------------------------------
class ProbabilityLearner(object):

    def __init__(self, p_step=.02, I_step=.2, k_step=.2, k_floor = 5e-4, k_celling = 20, i_floor = 2, i_celling = 1000):

        # Set up the parameter grids
        pl_grid = make_grid(.01, .99, p_step)
        self.pl_grid = pl_grid
        pr_grid = make_grid(.01, .99, p_step)
        self.pr_grid = pr_grid
        self.I_grid = make_grid(log(i_floor), log(i_celling), I_step)
        self.k_grid = make_grid(log(k_floor), log(k_celling), k_step)

        self._pl_size = pl_grid.size
        self._pr_size = pr_grid.size
        self._I_size = self.I_grid.size
        self._k_size = self.k_grid.size

        # Set up the transitional distribution on p
        I_trans = np.vectorize(I_trans_func)(*np.meshgrid(self.I_grid,
                                                          self.I_grid,
                                                          self.k_grid,
                                                          indexing="ij"))
        self._I_trans = I_trans / I_trans.sum(axis=0)

        pl_trans = np.vectorize(p_trans_func)(*np.meshgrid(self.pl_grid,
                                                          self.pl_grid,
                                                          self.I_grid,
                                                          indexing="ij"))
        pr_trans = np.vectorize(p_trans_func)(*np.meshgrid(self.pr_grid,
                                                          self.pr_grid,
                                                          self.I_grid,
                                                          indexing="ij"))
        self._pl_trans = pl_trans / pl_trans.sum(axis=0)
        self._pr_trans = pr_trans / pr_trans.sum(axis=0)
        # Initialize the learner and history
        self.reset()

        # Initialize the learner and history
        self.reset()

    @property
    def pl_hats(self):
        return np.atleast_1d(self._pl_hats)

    @property
    def pr_hats(self):
        return np.atleast_1d(self._pr_hats)

    @property
    def I_hats(self):
        return np.atleast_1d(self._I_hats)

    @property
    def k_hats(self):
        return np.atleast_1d(self._k_hats)

    @property
    def data(self):
        return np.atleast_1d(self._data)

    def fit(self, data):
        """Fit the model to a sequence of Bernoulli observations."""
        if np.isscalar(data):
            data = [data]
        for y in data:
            self._update(y)
            pI = self.pI
            self.p_dists.append(pI.sum(axis=1))
            self.I_dists.append(pI.sum(axis=0))
            self._p_hats.append(np.sum(self.p_dists[-1] * self.p_grid))
            self._I_hats.append(np.sum(self.I_dists[-1] * self.I_grid))
            self._data.append(y)

    def _update(self, y):
        pIk = self.pIk.copy()

        k_grid = self.k_grid
        I_grid = self.I_grid
        p_grid = self.p_grid

        Ip1gIk = self._I_trans
        pp1gpIp1 = self._p_trans

        for k in xrange(k_grid.size):

            # 1) Multiply pIk by Ip1gIk and integrate out I. This will give pIp1k
            pIp1k = np.zeros((p_grid.size, I_grid.size))
            for Ip1 in xrange(I_grid.size):
                for p in xrange(p_grid.size):
                    pIp1k[p, Ip1] = np.sum(Ip1gIk[Ip1, :, k] * pIk[p, :, k])

            # 2) Multiply pIp1k by pp1gpIp1 and integrate out p.
            pp1Ip1k = np.zeros((p_grid.size, I_grid.size))
            for Ip1 in xrange(I_grid.size):
                for pp1 in xrange(p_grid.size):
                    pp1Ip1k[pp1, Ip1] = np.sum(pIp1k[:, Ip1] *
                                            pp1gpIp1[pp1, :, Ip1].T)

            # 3) Place pp1Ip1k into pIk (belief that is carried to the next trial)
            pIk[:, :, k] = pp1Ip1k

        if reward:
            for k in xrange(k_grid.size):
                for p in xrange(p_grid.size):
                    pIk[p, :, k] *= p_grid[p]
        else:
            for k in xrange(k_grid.size):
                for p in xrange(p_grid.size):
                    pIk[p, :, k] *= 1 - p_grid[p]

        # Normalization
        pIk /= pIk.sum()

    def reset(self):
        """Reset the history of the learner."""
        # Initialize the joint distribution P(p, I, k)
        pI = np.ones((self._p_size, self._I_size))
        self.pI = pI / pI.sum()

        # Initialize the memory lists
        self.pl_dists = []
        self.pr_dists = []
        self.I_dists = []
        self._pl_hats = []
        self._pr_hats = []
        self._I_hats = []
        self._k_hats = []
        self._data = []

    def plot_history(self, ground_truth=None, **kwargs):
        """Plot the data and posterior means from the history."""
        blue, green = sns.color_palette("deep", n_colors=2)

        trials = np.arange(self.data.size)
        xlim = trials.min(), trials.max()

        f, (p_ax, I_ax) = plt.subplots(2, 1, sharex=True, **kwargs)
        p_ax.plot(trials, self.p_hats, c=blue)
        p_ax.scatter(trials, self.data, c=".25", alpha=.5, s=15)

        if ground_truth is not None:
            p_ax.plot(trials, ground_truth, c="dimgray", ls="--")
        p_ax.set_ylabel("$\hat p$", size=16)
        p_ax.set(xlim=xlim, ylim=(-.1, 1.1))

        I_ax.plot(trials, self.I_hats, c=green)
        I_ax.set_ylabel("$\hat I$", size=16)
        I_ax.set(ylim=(2, 10), xlabel=("Trial"))
        f.tight_layout()

    def plot_joint(self, cmap="BuGn"):
        """Plot the current joint distribution P(p, I)."""
        pal = sns.color_palette(cmap, 256)
        lc = pal[int(.7 * 256)]
        bg = pal[0]

        fig = plt.figure(figsize=(7, 7))
        gs = plt.GridSpec(6, 6)

        p_lim = self.p_grid.min(), self.p_grid.max()
        I_lim = self.I_grid.min(), self.I_grid.max()

        ax1 = fig.add_subplot(gs[1:, :-1])
        ax1.set(xlim=p_lim, ylim=I_lim)

        ax1.contourf(self.p_grid, self.I_grid, self.pI.T, 30, cmap=cmap)

        sns.axlabel("$p$", "$I$", size=16)

        ax2 = fig.add_subplot(gs[1:, -1], axis_bgcolor=bg)
        ax2.set(ylim=I_lim)
        ax2.plot(self.pI.sum(axis=0), self.I_grid, c=lc, lw=3)
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = fig.add_subplot(gs[0, :-1], axis_bgcolor=bg)
        ax3.set(xlim=p_lim)
        ax3.plot(self.p_grid, self.pI.sum(axis=1), c=lc, lw=3)
        ax3.set_xticks([])
        ax3.set_yticks([])