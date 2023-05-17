from numba import njit
import numpy as np
import time


@njit()
def J_mn_x(B, mx, my, nx, ny, dx, g, h):
    modB = np.sqrt(B[0] * B[0] + B[1] * B[1] + B[2] * B[2])
    modMN = np.sqrt(((mx - nx) * (mx - nx) + (my - ny) * (my - ny)))
    cos = (B[0] * (nx - mx) + B[1] * (ny - my)) / (modB * modMN)
    J = ((g ** 2) * (h ** 2) * (1 - 3 * (cos ** 2))) / ((modMN * dx) ** 3)
    return J


@njit()
def J_mn_y(B, mx, my, nx, ny, dx, g, h):
    return J_mn_x(B, mx, my, nx, ny, dx, g, h)


@njit()
def J_mn_z(B, mx, my, nx, ny, dx, g, h):
    return (-2.0) * J_mn_x(B, mx, my, nx, ny, dx, g, h)


@njit()
def J_mn(B, mx, my, nx, ny, dx, g, h):
    J = np.zeros(3)
    J[0] = J_mn_x(B, mx, my, nx, ny, dx, g, h)
    J[1] = J_mn_y(B, mx, my, nx, ny, dx, g, h)
    J[2] = J_mn_z(B, mx, my, nx, ny, dx, g, h)
    return J


@njit()
def is_in(i, j, N):
    if (i >= 0) and (i < N) and (j >= 0) and (j < N):
        return True
    return False


@njit()
def h_mn(S, B, N, mx, my, nx, ny, g, h, dx):
    h_c = np.zeros(3)
    if is_in(nx, ny, N):
        h_c = S[nx][ny] * J_mn(B, mx, my, nx, ny, dx, g, h)
    return h_c


@njit()
def count_hm_first(S, B, N, mx, my, g, h, dx):
    h_c = np.zeros(3)
    h_c += h_mn(S, B, N, mx, my, mx + 1, my + 0, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx + 0, my + 1, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx - 1, my + 0, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx + 0, my - 1, g, h, dx)
    return h_c


@njit()
def count_hm_second(S, B, N, mx, my, g, h, dx):
    h_c = np.zeros(3)
    h_c += h_mn(S, B, N, mx, my, mx + 1, my + 1, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx + 1, my - 1, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx - 1, my + 1, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx - 1, my - 1, g, h, dx)
    return h_c


@njit()
def count_hm_third(S, B, N, mx, my, g, h, dx):
    h_c = np.zeros(3)
    h_c += h_mn(S, B, N, mx, my, mx + 2, my + 0, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx + 0, my + 2, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx - 2, my + 0, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx + 0, my - 2, g, h, dx)
    return h_c


@njit()
def count_hm_forth(S, B, N, mx, my, g, h, dx):
    h_c = np.zeros(3)
    h_c += h_mn(S, B, N, mx, my, mx + 2, my + 1, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx + 2, my - 1, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx + 1, my + 2, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx - 1, my + 2, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx - 2, my + 1, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx - 2, my - 1, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx + 1, my - 2, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx - 1, my - 2, g, h, dx)
    return h_c


@njit()
def count_hm_fifth(S, B, N, mx, my, g, h, dx):
    h_c = np.zeros(3)
    h_c += h_mn(S, B, N, mx, my, mx + 2, my + 2, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx + 2, my - 2, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx - 2, my + 2, g, h, dx)
    h_c += h_mn(S, B, N, mx, my, mx - 2, my - 2, g, h, dx)
    return h_c


@njit()
def count_hm(S, B, N, mx, my, g, h, dx, n_eff):
    hm = np.zeros(3)
    if n_eff >= 0:
        hm += count_hm_first(S, B, N, mx, my, g, h, dx)
    if n_eff >= 1:
        hm += count_hm_second(S, B, N, mx, my, g, h, dx)
    if n_eff >= 2:
        hm += count_hm_third(S, B, N, mx, my, g, h, dx)
    if n_eff >= 3:
        hm += count_hm_forth(S, B, N, mx, my, g, h, dx)
    if n_eff >= 4:
        hm += count_hm_fifth(S, B, N, mx, my, g, h, dx)
    return hm


@njit()
def step(S, B, N, g, h, dx, dt, n_eff):
    S_new = np.zeros((N, N, 3))
    for i in range(N):
        for j in range(N):
            S_new[i, j] = S[i, j] + dt * np.cross(S[i, j], count_hm(S, B, N, i, j, g, h, dx, n_eff))
    return S_new


def count_mean(C, n_mean):
    C = np.array(C)
    C_new = []
    for i in range(len(C)):
        if i<n_mean:
            C_new.append(C[:i].mean())
        else:
            C_new.append(C[i-n_mean:i])
    return np.array(C_new)
