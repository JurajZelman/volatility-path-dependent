# Volatility is (mostly) path-dependent

Code for the paper [Volatility is (mostly) path-dependent - Guyon, Lekeufack (2022)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4174589) for the prediction of the daily realized volatility of the S&P500 index.

## Installation

- Clone the repository
- Install the `poetry` environment
- Run the code in `main.ipynb` notebook
- The `volatility` repository contains the methods for the calibration and estimation of the volatility via the path-dependent volatility model

## Abstract from the paper

We learn from data that volatility is mostly path-dependent: up to 90% of the variance of the implied volatility of equity indexes is explained endogenously by past index returns, and up to 65% for (noisy estimates of) future daily realized volatility. The path-dependency that we uncover is remarkably simple: a linear combination of a weighted sum of past daily returns and the square root of a weighted sum of past daily squared returns with different time-shifted power-law weights capturing both short and long memory. This simple model, which is homogeneous in volatility, is shown to consistently outperform existing models across equity indexes and train/test sets for both implied and realized volatility. It suggests a simple continuous-time path-dependent volatility (PDV) model that may be fed historical or risk-neutral parameters. The weights can be approximated by superpositions of exponential kernels to produce Markovian models. In particular, we propose a 4-factor Markovian PDV model which captures all the important stylized facts of volatility, produces very realistic price and (rough-like) volatility paths, and jointly fits SPX and VIX smiles remarkably well. We thus show that a continuous-time Markovian parametric stochastic volatility (actually, PDV) model can practically solve the joint SPX/VIX smile calibration problem. This article is dedicated to the memory of Peter Carr whose works on volatility modeling have been so inspiring to us.
