"""Posterior update for Gamma–Poisson emissions."""

from __future__ import annotations


def poisson_posterior(SS, prior):
    post = []
    a0 = prior['a0']
    b0 = prior['b0']
    for n, y_sum in zip(SS['n'], SS['y_sum']):
        aN = a0 + y_sum
        bN = b0 + n
        post.append({'a0': aN, 'b0': bN})
    return post

