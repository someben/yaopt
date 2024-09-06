#######
yaopt
#######                                      
                                         
.. image:: https://badge.fury.io/py/yaopt.png                                                                   
   :target: http://badge.fury.io/py/yaopt                                                   


Basic Options Pricing (in Python)
***********************************

“Oh cool. Probably a little easier than spinning up the QuantLib stack.” — `Wes McKinney <https://github.com/wesm>`_, creator of `Pandas <https://github.com/pydata/pandas>`_.


Features
==========

#. Option valuation w/ Black-Scholes, lattice (binomial tree), and Monte Carlo simulation models.                                  
#. Basic Greeks calculation (delta, theta, rho, vega, gamma) across each valuation model.
#. Discrete dividends support in the lattice (binomial tree) and Monte Carlo simulation models.
#. Early exercise (American options) support in Monte Carlo simulation through the Longstaff-Schwartz technique.
#. Minimal dependencies, just Numpy & SciPy.
#. Free software, released under the MIT license.

