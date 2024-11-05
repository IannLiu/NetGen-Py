import numpy as np
from scipy.integrate import solve_ivp
from typing import List


def odes(tp, bv: np.array, km: np.array, y1: np.array, y2: np.array):
    """
    An odes function for solving time-dynamics
    :param tp: time point
    :param bv: species variable
    :param km: the coefficients matrix
    :param y1: the species 1 variable matrix
    :param y2: the species 2 variable matrix
    :return:
    """
    bv = np.append(bv, 1)
    y1, y2 = bv[y1], bv[y2]

    return np.sum(km * y1 * y2, axis=1)


def solve_odes(t_span: tuple,
               ini_state: np.array,
               k_array: np.array,
               y_array: List[np.array],
               eval_interval: float = 1e-10):
    """
    Solving odes
    This function should receive parsed ode forms, Here is an example

    For the reactions like:
        A >> B + C with rate coefficient k1, A + C >> D + E with rate coefficient k2
    concentration list of A, B, C, D, E are : [0.04, 0.01, 0.02, 0.03, 0.005]
    the rate coefficients are: k = [10, 100]  | 10 for A >> B + C, 100 for A + C >> D + E
    for every species, the odes can be rewritten as: (stoichem_num * k) * y1 * y2
    for instance,
        for species A in A >> B + C,
            the stoichem_num is -1, k1 is 10, thus stoichem_num * k is -10
            reactant is A, the  first  species in concentration_list, thus y1 is 0, and y2 is -1 (a placeholder)
        for species A in A + C >> D + E,
            the stoichem_num is -1, k2 is 100, thus stoichem_num * k is -100
            reactant is A and C, the  first and third species in concentration_list, thus y1 is 0, and y2 is 2
    following this form, the matrix of   k_array, y1_array, y2_array is:
        k = [[-10, -100], [10, 0], [10, -100], [0, 100], [0, 100]]  # five species, two reaction,the shape is (5, 2)
        y1 = [[0, 0], [0, -1], [0, 0], [-1, 0], [-1, 0]]  # same with k
        y2 = [[-1, 2], [-1, -1], [-1, 2], [-1, 2], [-1, 2]]  # same with k
    y_array = [y1, y2]


    :param t_span: simulation time
    :param ini_state: initial state of species concentration
    :param k_array: the coefficients matrix
    :param y_array: the species variable matrix
    :param eval_interval: the evaluation interval for evaluating the edge species rate
    :return:
    """
    t_eval = np.arange(*t_span, eval_interval)
    params = (k_array, y_array[0], y_array[1])
    result_solve_ivp = solve_ivp(odes, t_span, ini_state, args=params,
                                 method='LSODA', t_eval=t_eval)

    return result_solve_ivp

