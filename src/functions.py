import numpy as np

def rosenbrock(x):
    result = np.sum(100*(x[1:]-x[:-1]**2)**2 + (x[:-1]-1)**2)
    return result

def rastrigin(x):
    D = len(x) 
    result = 10*D + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    return result

def griewank(x):
    sum_term = np.sum(x**2)/4000
    prod_term = np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))))
    result = 1 + sum_term - prod_term
    return result