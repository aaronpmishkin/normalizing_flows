# @Author: aaronmishkin
# @Date:   18-11-25
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-11-25

import torch

import lib.root_finding.newton as newton


# Solve a simple root finding problem:


quad = lambda x: x**2 - 16
deriv_quad = lambda x: 2 * x
x_0 = 10

x_root = newton.newtons_method(quad, deriv_quad, x_0)
print(x_root)

quad = lambda x: x**2 - 16
deriv_quad = lambda x: 2 * x
x_0 = 10

x_root = newton.newtons_method(quad, deriv_quad, x_0)
