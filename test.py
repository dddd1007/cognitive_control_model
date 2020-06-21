class test:
    def __init__(self, name):
        self.name = name

    def print_str(self, inputstr):
        print(inputstr)

testor = test("testor")

testor.print_str("Hello World")


import numpy as np
import pymc3 as pm

with pm.Model() as test_model:
    x = pm.Normal('x', mu = 0, sigma = 1)