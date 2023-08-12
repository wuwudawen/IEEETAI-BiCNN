import numpy as np
import cvxpy as cp

def R(x, y, R_matrix):
    return x.T @ R_matrix @ y

def C(x, y, C_matrix):
    return x.T @ C_matrix @ y

def reg_r(x, y, R_matrix):
    max_val = -np.inf
    for i in range(len(x)):
        e_i = np.zeros_like(x)
        e_i[i] = 1
        max_val = max(max_val, R(e_i, y, R_matrix))
    return max_val - R(x, y, R_matrix)

def reg_c(x, y, C_matrix):
    max_val = -np.inf
    for j in range(len(y)):
        e_j = np.zeros_like(y)
        e_j[j] = 1
        max_val = max(max_val, C(x, e_j, C_matrix))
    return max_val - C(x, y, C_matrix)

def LP_1(y0, R_matrix, C_matrix):
    n = R_matrix.shape[0]

    max_val_1 = -np.inf
    for i in range(n):
        e_i = np.zeros((n, ))
        e_i[i] = 1
        max_val_1 = max(max_val_1, R(e_i, y0, R_matrix))

    x = cp.Variable(n)

    objective = cp.Minimize(max_val_1 - x@(R_matrix@y0))
    constraints = [max_val_1-x@(R_matrix@y0) >= cp.max(x@C_matrix)-x@(C_matrix@y0),
            x @ np.ones((n, )) == 1,
            x >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return x.value

def LP_2(x0, R_matrix, C_matrix):
    n = R_matrix.shape[0]

    max_val_2 = -np.inf
    for j in range(n):
        e_j = np.zeros((n, ))
        e_j[j] = 1
        max_val_2 = max(max_val_2, C(x0, e_j, C_matrix))

    y = cp.Variable(n)

    objective = cp.Minimize(max_val_2 - (x0@C_matrix)@y)
    constraints = [max_val_2-(x0@C_matrix)@y >= cp.max(R_matrix@y)-(x0@R_matrix)@y,
            y @ np.ones((n, )) == 1,
            y >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return y.value

def regreat_equalization(x0, y0, R_matrix, C_matrix):
    if reg_r(x0, y0, R_matrix) >= reg_c(x0, y0, C_matrix):
        x = LP_1(y0, R_matrix, C_matrix)
        return (x, y0)
    else:
        y = LP_2(x0, R_matrix, C_matrix)
        return (x0, y)

def find_max_indices(vector):
    max_val = np.max(vector)
    max_indices = np.argwhere(vector == max_val)

    return max_indices.flatten()


def primal_LP(x, y, R_matrix, C_matrix):
    n = R_matrix.shape[0]

    gamma = cp.Variable()
    xp = cp.Variable(n)
    yp = cp.Variable(n)

    Br = find_max_indices(R_matrix@y)
    Bc = find_max_indices(x@C_matrix)

    objective = cp.Minimize(gamma)
    constraint1 = [gamma >= (np.eye(n)[i]@R_matrix)@yp - (x@R_matrix)@yp - xp@(R_matrix@y) + (x@R_matrix)@y for i in Br]
    constraint2 = [gamma >= xp@(C_matrix@np.eye(n)[j]) - xp@(C_matrix@y) - (x@C_matrix)@yp + (x@C_matrix)@y for j in Bc]
    constraint3 = [xp @ np.ones((n, )) == 1, xp >= 0, yp @ np.ones((n, )) == 1, yp >= 0]
    constraints = constraint1+constraint2+constraint3

    prob = cp.Problem(objective, constraints)
    prob.solve()
    return gamma.value, xp.value, yp.value

def g(x, y, R_matrix, C_matrix):
    return max(reg_r(x, y, R_matrix), reg_c(x, y, C_matrix))

def descent_phase(x, y, R_matrix, C_matrix, delta):
    iter = 0
    while True:
        iter += 1
        x, y = regreat_equalization(x, y, R_matrix, C_matrix)
        gamma, xp, yp = primal_LP(x, y, R_matrix, C_matrix)
        if gamma-g(x, y, R_matrix, C_matrix)>=-delta:
            xs, ys = x, y
            return xs, ys, iter
        if iter>=50:
            xs, ys = x, y
            return xs, ys, iter
        x = (1-delta/(delta+2))*x + (delta/(delta+2))*xp
        y = (1-delta/(delta+2))*y + (delta/(delta+2))*yp

def dual_LP(x, y, R_matrix, C_matrix):
    n = R_matrix.shape[0]
    Br = find_max_indices(R_matrix@y)
    Bc = find_max_indices(x@C_matrix)
    Br_index = np.arange(len(Br))
    Bc_index = np.arange(len(Bc))

    a = cp.Variable()
    b = cp.Variable()
    p = cp.Variable(Br.shape[0])
    q = cp.Variable(Bc.shape[0])

    P = cp.sum(p)
    Q = cp.sum(q)

    objective = cp.Maximize(P*(x@R_matrix@y)+Q*(x@C_matrix@y)+a+b)
    c1 = [p>=0]
    c2 = [q>=0]
    c3 = [P+Q==1]
    c4 = []
    for k in range(n):
        temp = P*(-np.eye(n)[k]@R_matrix@y)
        for j in Bc_index:
            temp = temp + (-np.eye(n)[k]@C_matrix@y+C_matrix[k,Bc[j]])*q[j]
        c4 = c4 + [a <= temp]
    c5 = []
    for l in range(n):
        temp = Q*(-x@C_matrix@np.eye(n)[l])
        for i in Br_index:
            temp = temp + (-x@R_matrix@np.eye(n)[l]+R_matrix[Br[i],l])*p[i]
        c5 = c5 + [b <= temp]
    constraints = c1+c2+c3+c4+c5
    prob = cp.Problem(objective, constraints)
    prob.solve()

    p = p.value
    q = q.value

    w, z = np.zeros((n, )), np.zeros((n, ))
    for i in Br_index:
        w[Br[i]] = p[i]/np.sum(p)
    for j in Bc_index:
        z[Bc[j]] = q[j]/np.sum(q)

    lamba = w@R_matrix@z - x@R_matrix@z
    mu = w@C_matrix@z - w@C_matrix@y
    return w, z, lamba, mu

def strategy_construction(xs, ys, w, z, lamba, mu, R_matrix, C_matrix):
    if min(lamba, mu) <= 0.5:
        return xs, ys
    if min(lamba, mu) >= 2/3:
        return w, z
    if min(lamba, mu) > 1/2 and max(lamba, mu)<=2/3:
        return xs, ys
    if lamba>=mu:
        xt, yt = (1/(1+lamba-mu))*w + (lamba-mu/(1+lamba-mu))*xs, z
        if g(xt, yt, R_matrix, C_matrix)<=g(xs, ys, R_matrix, C_matrix):
            return xt, yt
        else:
            return xs, ys
    if lamba<mu:
        xt, yt = w, (1/(1+mu-lamba))*z + (mu-lamba/(1+mu-lamba))*ys
        if g(xt, yt, R_matrix, C_matrix)<=g(xs, ys, R_matrix, C_matrix):
            return xt, yt
        else:
            return xs, ys

def improved_strategy_construction(xs, ys, w, z, lamba, mu, R_matrix, C_matrix):
    n = R_matrix.shape[0]

    if min(lamba, mu) <= 0.5:
        return xs, ys
    if min(lamba, mu) >= 2/3:
        return w, z
    if min(lamba, mu) > 1/2 and max(lamba, mu)<=2/3:
        return xs, ys
    if 0.5<lamba<=2/3<mu:
        y_hat = 0.5*ys+0.5*z
        w_hat = find_max_indices(R_matrix@y_hat)
        w_hat = np.eye(n)[w_hat[0]]
        tr = w_hat@R_matrix@y_hat - w@R_matrix@y_hat
        vr = w@R_matrix@ys - w_hat@R_matrix@ys
        mu_hat = w_hat@C_matrix@z - w_hat@C_matrix@ys
        if vr+tr>=(mu-lamba)/2 and mu_hat>=mu-vr-tr:
            p = (2*(vr+tr)-(mu-lamba))/(2*(vr+tr))
            if g(p*w+(1-p)*w_hat, z, R_matrix, C_matrix)<=g(xs, ys, R_matrix, C_matrix):
                return p*w+(1-p)*w_hat, z
            else:
                return xs, ys
        else:
            q = (1-0.5*mu-tr)/(1+0.5*mu-lamba-tr)
            if g(w, (1-q)*y_hat+q*z, R_matrix, C_matrix)<=g(xs, ys, R_matrix, C_matrix):
                return w, (1-q)*y_hat+q*z
            else:
                return xs, ys

    if 0.5<mu<=2/3<lamba:
        x_hat = 0.5*xs+0.5*w
        z_hat = find_max_indices(x_hat@C_matrix)
        z_hat = np.eye(n)[z_hat[0]]
        tc = x_hat@C_matrix@z_hat - x_hat@C_matrix@z
        vc = xs@C_matrix@z - xs@C_matrix@z_hat
        lamba_hat = w@R_matrix@z_hat - xs@R_matrix@z_hat
        if vc+tc>=(lamba-mu)/2 and lamba_hat>=lamba-vc-tc:
            p = (2*(vc+tc)-(lamba-mu))/(2*(vc+tc))
            if g(w, p*z+(1-p)*z_hat, R_matrix, C_matrix)<=g(xs, ys, R_matrix, C_matrix):
                return w, p*z+(1-p)*z_hat
            else:
                return xs, ys
        else:
            q = (1-0.5*lamba-tc)/(1+0.5*lamba-mu-tc)
            if g((1-q)*x_hat+q*w, z, R_matrix, C_matrix)<=g(xs, ys, R_matrix, C_matrix):
                return w, (1-q)*y_hat+q*z
            else:
                return xs, ys

def TS(INSTANCE, VALUE, warm_start=False, x_init=None, y_init=None):
    A, B = INSTANCE[:,0,:,:], INSTANCE[:,1,:,:]
    batch_size = A.shape[0]
    gs = A.shape[1]

    A_norm, B_norm = np.zeros_like(A), np.zeros_like(B)
    for i in range(batch_size):
        A_norm[i], B_norm[i] = normalize_matrices(A[i], B[i])
    x_TS, y_TS = np.zeros((batch_size, gs)), np.zeros((batch_size, gs))

    if warm_start is False:
        x_init = np.ones((batch_size, gs))/gs
        y_init = np.ones((batch_size, gs))/gs
    else:
        x_init = x_init
        y_init = y_init

    time_TS = np.zeros((batch_size, ))
    iter_TS = np.zeros((batch_size, ))
    for i in range(batch_size):
        start_time = time.time()

        xs, ys, iter = descent_phase(x_init[i], y_init[i], A_norm[i], B_norm[i], 0.2)
        w, z, lamba, mu = dual_LP(xs, ys, A_norm[i], B_norm[i])
        x_TS[i], y_TS[i] = strategy_construction(xs, ys, w, z, lamba, mu, A_norm[i], B_norm[i])

        time_TS[i] = time.time()-start_time
        iter_TS[i] = iter
        # print(i)
    return x_TS, y_TS, time_TS, iter_TS

def DFM(INSTANCE, VALUE, warm_start=False, x_init=None, y_init=None):
    A, B = INSTANCE[:,0,:,:], INSTANCE[:,1,:,:]
    batch_size = A.shape[0]
    gs = A.shape[1]

    A_norm, B_norm = np.zeros_like(A), np.zeros_like(B)
    for i in range(batch_size):
        A_norm[i], B_norm[i] = normalize_matrices(A[i], B[i])
    x_DFM, y_DFM = np.zeros((batch_size, gs)), np.zeros((batch_size, gs))

    if warm_start is False:
        x_init = np.ones((batch_size, gs))/gs
        y_init = np.ones((batch_size, gs))/gs
    else:
        x_init = x_init
        y_init = y_init

    time_DFM = np.zeros((batch_size, ))
    iter_DFM = np.zeros((batch_size, ))
    for i in range(batch_size):
        start_time = time.time()

        xs, ys, iter = descent_phase(x_init[i], y_init[i], A_norm[i], B_norm[i], 0.2)
        w, z, lamba, mu = dual_LP(xs, ys, A_norm[i], B_norm[i])
        x_DFM[i], y_DFM[i] = improved_strategy_construction(xs, ys, w, z, lamba, mu, A_norm[i], B_norm[i])

        time_DFM[i] = time.time()-start_time
        iter_DFM[i] = iter
        # print(i)
    return x_DFM, y_DFM, time_DFM, iter_DFM
