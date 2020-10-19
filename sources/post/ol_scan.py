import glob, os, sys
import numpy as np
import scipy.linalg as la
import scipy.sparse as spa
from multiprocessing import Pool
import overlaps as ol

import mkl
mkl.set_num_threads(1)

def nrms_by_sep(psi):
  """Reduces psi by modding out one of the positions. Only the distance
  between the B's counts.

  Input: psi of shape [num_sites, num_sites, ...]
  Output: [num_sites, ...]

  Axis 1 of the output is the separation between the two B tensors.
  output[0, ...] is always zero, since the tensors never overlap.
  """
  shp = psi.shape
  psi = psi.reshape((shp[0], shp[1], -1))
  sqs = np.zeros((psi.shape[2], psi.shape[0]))
  for i in range(psi.shape[2]):
    psi_i = np.nan_to_num(psi[:,:,i])
    for j in range(psi_i.shape[0]):
      sqs_ij = np.abs(psi_i[j, j:])**2
      sqs[i, :len(sqs_ij)] += sqs_ij
  sqs = sqs.transpose()
  sqs = sqs.reshape((sqs.shape[0], *shp[2:]))
  return np.sqrt(sqs)


def state_ol_1(state_data):
    t, fn = state_data
    print("Starting:", t)
    state = ol.load_state(fn, ham=ham)
    
    psi_fn_comp = path + "psi_comp_{}_nex{}_t{}.npz".format(sec_lbl, num_ex, t)
    try:
      data = np.load(psi_fn_comp)  # npz
      psi = data["psi"]
      print("Loaded psi:", t)
    except IOError:
      # each step takes O(N) time
      psi = ol.overlap_1p_components_triv(state, num_ex, verbose=False)
      np.savez_compressed(psi_fn_comp, psi=psi)

    nrms_by_ex = la.norm(np.nan_to_num(psi), axis=1)
    print("Done:", t, la.norm(nrms_by_ex))
    return nrms_by_ex


def state_ol_2(state_data):
    t, fn = state_data
    print("Starting:", t)
    state = ol.load_state(fn, ham=ham)
    
    #psi_fn = path + "psi_{}_nex{}_np{}_t{}.npy".format(sec_lbl, num_ex, len(ps), t)
    psi_fn_comp = path + "psi_comp_{}_nex{}_np{}_t{}.npz".format(sec_lbl, num_ex, len(ps), t)
    try:
      #psi = np.load(psi_fn)
      #exlbls = np.load(path + "exlbls_{}_nex{}_np{}.npy".format(sec_lbl, num_ex, len(ps)), allow_pickle=True)
      #np.savez_compressed(psi_fn_comp, psi=psi, exlbls=exlbls)

      data = np.load(psi_fn_comp)  # npz
      psi = data["psi"]
      exlbls = data["exlbls"]
      print("Loaded psi:", t)
    except IOError:
      # each step takes O(N^2) time
      psi, _, exlbls, sL, sR, BLs, BRs = ol.overlap_2p_components_multip(
          state, vac_mid, ps=ps, num_ex=num_ex, ortho=True,
          brute=True, force_pseudo=force_pseudo, verbose=False,
          return_prep_data=True)
      print(psi.shape)
      print(exlbls)
      #np.save(psi_fn, psi)
      np.savez_compressed(psi_fn_comp, psi=psi, exlbls=exlbls, BLs=BLs, BRs=BRs)

    nrms_allseps = nrms_by_sep(psi)
    np.save(path + "norms_allsep_{}_nex{}_np{}_t{}.npy".format(sec_lbl, num_ex, len(ps), t), nrms_allseps)

    nrms = la.norm(nrms_allseps, axis=0)
    nrms_sep = la.norm(nrms_allseps[d_min:], axis=0)
    print("Done:", t, la.norm(nrms), la.norm(nrms_sep))
    return exlbls, nrms, nrms_sep


os.environ["OMP_NUM_THREADS"] = "1"

sec_lbl = sys.argv[1]

interval = int(sys.argv[2])
t_max = int(sys.argv[3])

num_procs = int(sys.argv[4])

path = "./"
basedir = "../"
ham = ol.load_ham(path)

if sec_lbl == "2p":
  vac_mid = ol.tdvp_u.EvoMPS_TDVP_Uniform.from_file(
      basedir + "vac_out_uniform.npy", ol.load_ham(basedir))
  force_pseudo = True
  ps = [0.0, -0.6]
  num_ex = 3
  d_min = 40
  state_ol = state_ol_2
elif sec_lbl == "2k":
  vac_mid = ol.tdvp_u.EvoMPS_TDVP_Uniform.from_file(
      basedir + "vac_in_uniform.npy", ol.load_ham(basedir))
  force_pseudo = False
  ps = [0.0, -0.4, 0.4]
  num_ex = 1
  d_min = 15
  state_ol = state_ol_2
elif sec_lbl == "1p":
  num_ex = 64
  state_ol = state_ol_1
else:
  raise ValueError("Invalid sector label: {}".format(sec_lbl))





if __name__ == '__main__':
  pool = Pool(num_procs)

  sds = [(t, fn) for t, fn in ol.scan_states(path, interval, t_max) if os.path.isfile(fn)]
  ts = [t for t, fn in sds]
  print(ts)
  data = pool.map(state_ol, sds)

  if state_ol == state_ol_2:
      exlbls, nrms, nrms_sep = list(zip(*data))

      np.save(path + "norms_{}_nex{}_np{}.npy".format(sec_lbl, num_ex, len(ps)), nrms)
      np.save(
        "norms_{}_nex{}_np{}{}.npy".format(
          sec_lbl, num_ex, len(ps), "_sep{}".format(d_min)), nrms_sep)
      np.save(path + "norms_{}_nex{}_np{}.npy".format(sec_lbl, num_ex, len(ps)), nrms)
      np.save(path + "ts_{}_nex{}_np{}.npy".format(sec_lbl, num_ex, len(ps)), ts)
      np.save(path + "exlbls_{}_nex{}_np{}.npy".format(sec_lbl, num_ex, len(ps)), exlbls[0])
  elif state_ol == state_ol_1:
      nrms_by_exs = data
      
      np.save(path + "ts_{}_nex{}.npy".format(sec_lbl, num_ex), ts)
      np.save(path + "norms_{}_nex{}.npy".format(sec_lbl, num_ex), nrms_by_exs)
