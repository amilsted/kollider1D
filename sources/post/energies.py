import os
import json
from multiprocessing import Pool
import numpy as np
import scipy.linalg as la
import evoMPS.mps_sandwich as mps_s
import evoMPS.tdvp_uniform as tdvp

import mkl
mkl.set_num_threads(1)

E = np.eye(2)
Sx = np.array(
  [[0, 1],
   [1, 0]])
Sy = 1.j * np.array(
  [[0, -1],
   [1, 0]])
Sz = np.array(
  [[1, 0],
   [0, -1]])


def get_ham(J, hx, hz=0.0, J2=0.0, J3=0.0, J4=0.0, J5=0.0, hzx=0.0):
  ham = -(J * np.kron(Sz, Sz)
          + J3 * np.kron(Sx, Sx)
          + hx * np.kron(Sx, np.eye(2))
          + hz * np.kron(Sz, np.eye(2))
          + hzx * (np.kron(Sz, Sx) + np.kron(Sx, Sz))
          ).reshape(2, 2, 2, 2)
  if J2 != 0.0 or J4 != 0.0 or J5 !=0.0:
    ham = np.kron(ham.reshape(4, 4), np.eye(2))
    ham += -J2 * np.kron(np.kron(Sz, np.eye(2)), Sz)
    ham += -J4 * np.kron(np.kron(Sz, Sz), Sx)
    ham += -J5 * np.kron(np.kron(Sx, Sz), Sz)
    ham = ham.reshape(2, 2, 2, 2, 2, 2)
  return ham


def get_ham_uniform(J, hx, hz=0.0, J2=0.0, J3=0.0, J4=0.0, J5=0.0, hzx=0.0):
  ham = -(J * np.kron(Sz, Sz)
          + J3 * np.kron(Sx, Sx)
          + hx/2 * (np.kron(Sx, np.eye(2)) + np.kron(np.eye(2), Sx))
          + hz/2 * (np.kron(Sz, np.eye(2)) + np.kron(np.eye(2), Sz))
          + hzx * (np.kron(Sz, Sx) + np.kron(Sx, Sz))
          ).reshape(2, 2, 2, 2)
  if J2 != 0.0 or J4 != 0.0 or J5 !=0.0:
    ham = 0.5 * (np.kron(ham.reshape(4, 4), np.eye(2)) + np.kron(np.eye(2), ham.reshape(4, 4)))
    ham += -J2 * np.kron(np.kron(Sz, np.eye(2)), Sz)
    ham += -J4 * np.kron(np.kron(Sz, Sz), Sx)
    ham += -J5 * np.kron(np.kron(Sx, Sz), Sz)
    ham = ham.reshape(2, 2, 2, 2, 2, 2)
  return ham


def dense_to_mpo(op, eps=1e-12):
    """Output order [mL,mR,p_out,p_in]
    """
    N = len(op.shape) // 2
    perm = [(N if j % 2 == 0 else 1) + j // 2 - 1 for j in range(1,2*N+1)]
    op = op.transpose(perm)
    op = op.reshape([1, *op.shape])
    right = op
    mpo = []
    while N > 1:
        right_mat = right.reshape((np.prod(right.shape[:3]), np.prod(right.shape[3:])))
        U, S, Vh = la.svd(right_mat, full_matrices=False)
        i_trunc = np.argmax(S < eps)
        print("error: ", la.norm(S[i_trunc:]))
        U = U[:,:i_trunc]
        S = S[:i_trunc]
        Vh = Vh[:i_trunc, :]
        mpo.append(np.transpose(U.reshape([*right.shape[:3], U.shape[1]]), (0,3,1,2)))
        print("Bond dim: {}".format(U.shape[1]))
        N -= 1
        right = (np.diag(S) @ Vh).reshape([len(S), *(right.shape[3:])])

    right = np.transpose(np.reshape(right, [*(right.shape), 1]), (0,3,1,2))
    mpo.append(right)
    return mpo


def state_fn(dt, t):
    ind = int(t / dt) - 1
    return "state_evo_step{}.npy".format(ind)


def load_state(filename, ham=None, do_update=True):
    s = mps_s.EvoMPS_MPS_Sandwich.from_file(filename)
    if ham is not None:
        s.update(restore_CF=False)
        #s = tdvp_s.EvoMPS_TDVP_Sandwich.from_mps(s, ham, [ham] * (s.N + 1), ham)
        s.uni_l = tdvp.EvoMPS_TDVP_Uniform.from_mps(s.uni_l, ham)
        s.uni_r = tdvp.EvoMPS_TDVP_Uniform.from_mps(s.uni_r, ham)
    if do_update:
        s.update()
        if ham is not None:
            #s.uni_l.calc_lr(rescale=False)
            s.uni_l.calc_C()
            s.uni_l.calc_K()
            #s.uni_r.update(restore_CF=False)
            s.uni_r.calc_C()
            s.uni_r.calc_K()
    return s


def compute_energies(state_data):
    t, fn = state_data
    try:
        state = load_state(fn, ham=None)
    except:
        print("Error loading:", t, fn)
        blank = np.full((state_params["N"] + 1,), np.NaN)
        return blank, blank

    ecur = np.array([state.expect_MPO(p_mpo,i).real for i in range(state.N+1)])
    en = np.array([state.expect_MPO(h_mpo,i).real for i in range(state.N+1)])
    print(t, fn, sum(en))
    return en, ecur


with open("state_params.json", "r") as f:
    state_params = json.load(f)
print(state_params)


with open("evo_params.json", "r") as f:
    evo_params = json.load(f)
print(evo_params)
dt = evo_params["dt"]
t_end = evo_params["t_end"]


ham_u = get_ham_uniform(state_params["J"], state_params["hx"], state_params["hz"], state_params["J2"], state_params["J3"],
             state_params["J4"], state_params["J5"], state_params["hzx"])


ham_sites = len(ham_u.shape) // 2
ham_mat = ham_u.reshape(2**ham_sites,2**ham_sites)
h1 = np.kron(ham_mat, E)
h2 = np.kron(E, ham_mat)
p_mat = -1.0j * (h1 @ h2 - h2 @ h1)
p = p_mat.reshape([2]*2*(ham_sites+1))
p_mpo = dense_to_mpo(p)
h_mpo = dense_to_mpo(ham_u)

num_procs = 8

if __name__ == '__main__':
  pool = Pool(num_procs)

  ts = np.arange(0, int(t_end) + 1)
  fns = ["../initial_state.npy"] + [state_fn(dt, t) for t in ts[1:]]

  for i in range(len(fns)):
    if not os.path.isfile(fns[i]):
      fns = fns[:i]
      ts = ts[:i]
      break

  print("Data up to t =", ts[-1])
  data = pool.map(compute_energies, zip(ts, fns))
  e_densities, e_currents = list(zip(*data))
    
  np.save("energies_evo.npy", np.array(e_densities))
  np.save("encurrs_evo.npy", np.array(e_currents))
