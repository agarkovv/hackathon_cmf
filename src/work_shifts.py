import numpy as np
from scipy.optimize import linprog


class WrongCalculateOfIndicatorMatrix(Exception):
    pass


tests = [[3, 4, 4, 4, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1], [1, 5, 3, 1, 2, 1, 3, 5, 6, 3, 1]]
shift_hours = [4, 5, 6, 7, 8]


def make_work_shifts(partner_bounds):
    n = len(partner_bounds)
    weights = []
    for h in shift_hours:
        weights += [h] * (n - h + 1)

    indicator_matrix = []
    for i in range(n):
        ind_row = []
        for h in shift_hours:
            for j in range(n - h + 1):
                ind_row.append(int(j <= i <= h + j - 1))
        indicator_matrix.append(ind_row)

    indicator_matrix = np.array(indicator_matrix)
    partner_bounds = np.array(partner_bounds)
    weights = np.array(weights)

    if (indicator_matrix.sum(axis=0) != weights).any():
        raise WrongCalculateOfIndicatorMatrix

    res = linprog(weights, A_ub=-indicator_matrix, b_ub=-partner_bounds, bounds=(0, None), method='simplex')
    x = np.rint(res.x)

    shifts = {}
    i = 0
    for h in shift_hours:
        shifts[h] = []
        for j in range(n - h + 1):
            shifts[h] += [j] * int(x[i])
            i += 1
    return shifts, indicator_matrix @ x,


for test in tests:
    work_shifts, partner_cnt = make_work_shifts(test)
    print("test:", test)
    print(work_shifts, partner_cnt, sep="\n")
