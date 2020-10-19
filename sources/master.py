"""
Creates a localized kink wavepacket and evolve it.
"""

import os
import sys
import copy
import pickle
import json
import numpy as np
import scipy as sp

import evoMPS.tdvp_uniform as tdvp
import evoMPS.tdvp_sandwich as tdvp_s
import evoMPS.dynamics as dyn
import evoMPS.split_step as ss
import evoMPS.tdvp_common as tm
import evoMPS.matmul as mm

#import mkl
#mkl.set_num_threads(1)

Sx = sp.array(
  [[0, 1],
   [1, 0]])


Sy = 1.j * sp.array(
  [[0, -1],
   [1, 0]])


Sz = sp.array(
  [[1, 0],
   [0, -1]])


def get_ham(J, hx, hz=0.0, J2=0.0, J3=0.0, J4=0.0, J5=0.0, hzx=0.0):
  ham = -(J * sp.kron(Sz, Sz)
          + J3 * sp.kron(Sx, Sx)
          + hx * sp.kron(Sx, sp.eye(2))
          + hz * sp.kron(Sz, sp.eye(2))
          + hzx * (sp.kron(Sz, Sx) + sp.kron(Sx, Sz))
          ).reshape(2, 2, 2, 2)
  if J2 != 0.0 or J4 != 0.0 or J5 !=0.0:
    ham = sp.kron(ham.reshape(4, 4), sp.eye(2))
    ham += -J2 * sp.kron(sp.kron(Sz, sp.eye(2)), Sz)
    ham += -J4 * sp.kron(sp.kron(Sz, Sz), Sx)
    ham += -J5 * sp.kron(sp.kron(Sx, Sz), Sz)
    ham = ham.reshape(2, 2, 2, 2, 2, 2)
  return ham


def cbf(s, i, **kwargs):
  h = s.h_expect.real
  row = [str(i)]
  row.append("%.15g" % h)

  """
  Compute expectation values!
  """
  exSzs = []
  for k in range(s.L):
    exSzs.append("%.3g" % s.expect_1s(Sz, k=k).real)
  row += exSzs

  row.append("%.6g" % s.eta.real)
  row.append(str(kwargs))

  print("\t".join(row))


def gaussian_packet_coeff(N, sigma, midpt, p=0):
  return np.array([np.exp(1.j*p*j) * np.exp(-(j - midpt)**2/sigma**2) for j in range(N+1)])


def top_excitations(sL, sR, p=0.0, phase_align=False, force_pseudo=False):
    ev, eV = sL.excite_top_nontriv_brute(
        sR, p, return_eigenvectors=True, phase_align=phase_align,
        force_pseudo=force_pseudo)

    xs = [
    eV[:,i].reshape((sL.D, (sL.q - 1) * sL.D))
        for i in range(eV.shape[1])]

    Bs = [
        sL.get_B_from_x(
          x, sR.Vsh[0], sL.l_sqrt_i[0], sR.r_sqrt_i[0])
        for x in xs]

    return ev, Bs


def scattering_state(N, vacL, vacR, fvac, BL, BR, coeffL, coeffR, zero_threshold=1e-16, N_centre=None):
    sw = tdvp_s.EvoMPS_MPS_Sandwich(N, copy.deepcopy(vacL), copy.deepcopy(vacR), update_bulks=False)
    if N_centre is not None:
        sw.N_centre = N_centre
    
    # we assume coefficients only go to zero at the ends

    d = sw.A[1].shape[0]
    D = sw.A[1].shape[1]  #assume uniform D

    As = [None] * (N+1)
    Ds = copy.copy(sw.D)
    
    len_L = len(coeffL)
    len_R = len(coeffR)
    if N < len_L + len_R + 1:
        raise ValueError("System size must be more than 2 times the packet length.")

    j = 1
    while abs(coeffL[j]) < zero_threshold and j <= len_L:
        As[j] = vacL.A[0]
        j += 1

    As[j] = np.zeros((d, D, 2 * D), dtype=sw.A[1].dtype)
    As[j][:,:,:D] = BL * coeffL[j]
    As[j][:,:,D:] = vacL.A[0]
    Ds[j] = 2 * D
    j += 1

    while j < len_L-1 and abs(coeffL[j + 1]) >= zero_threshold:
        As[j] = np.zeros((d, 2 * D, 2 * D), dtype=sw.A[1].dtype)
        As[j][:,:D,:D] = fvac.A[0]
        As[j][:,D:,D:] = vacL.A[0]
        As[j][:,D:,:D] = BL * coeffL[j]
        Ds[j] = 2*D
        j += 1

    As[j] = np.zeros((d, 2*D, D), dtype=sw.A[1].dtype)
    As[j][:,:D,:] = fvac.A[0]
    As[j][:,D:,:] = BL * coeffL[j]
    j += 1

    while j <= N-len_R:
        As[j] = fvac.A[0]
        j += 1
        
    k = 0
    while abs(coeffR[k]) < zero_threshold and k <= len_R:
        As[j] = fvac.A[0]
        j += 1
        k += 1
        
    As[j] = np.zeros((d, D, 2 * D), dtype=sw.A[1].dtype)
    As[j][:,:,:D] = BR * coeffR[k]
    As[j][:,:,D:] = fvac.A[0]
    Ds[j] = 2 * D
    j += 1
    k += 1

    while k < len_R-1 and abs(coeffR[k + 1]) >= zero_threshold:
        As[j] = np.zeros((d, 2 * D, 2 * D), dtype=sw.A[1].dtype)
        As[j][:,:D,:D] = vacR.A[0]
        As[j][:,D:,D:] = fvac.A[0]
        As[j][:,D:,:D] = BR * coeffR[k]
        Ds[j] = 2*D
        j += 1
        k += 1

    As[j] = np.zeros((d, 2*D, D), dtype=sw.A[1].dtype)
    As[j][:,:D,:] = vacR.A[0]
    As[j][:,D:,:] = BR * coeffR[k]
    j += 1
    k += 1

    while j <= N:
        As[j] = vacR.A[0]
        j += 1

    if not np.all(Ds == sw.D):
        sw.D = Ds
        sw._init_arrays()
    for j in range(1, N+1):
        sw.A[j][:] = As[j]

    sw.update(restore_CF=True)  # this will also normalize the wavefunction
    return sw


def rgf_violation(B, A, r):
    return np.linalg.norm(tm.eps_r_noop(r, B, A))


def hamiltonian_elements(B1, B2, AL, AR, lL, rR, KL, KR, h_nn, d_max):
    B1_viol = rgf_violation(B1, AR, rR)
    B2_viol = rgf_violation(B2, AR, rR)
    if B1_viol > 1e-12:
        raise ValueError("Gauge-fixing condition not satisfied for B1! Violation: {}".format(B1_viol))
    if B2_viol > 1e-12:
        raise ValueError("Gauge-fixing condition not satisfied for B2! Violation: {}".format(B2_viol))
        
    print(mm.adot(lL, tm.eps_r_noop(rR, B1, B2)))
    
    matels = []
    if d_max >= 0:
        # <lL|(B1;B2)|KR>
        x = tm.eps_l_noop(lL, B1, B2)
        res1 = mm.adot(x, KR)
        
        # <KL|(B1;B2)|rR>
        x = tm.eps_l_noop(KL, B1, B2)
        res2 = mm.adot_noconj(x, rR)  # Do not conjugate KL contribution
        
        # <lL|h(AL,B1;AL,B2)|rR>
        x = tm.eps_r_op_2s_A(rR, AL, B1, AL, B2, h_nn)
        res3 = mm.adot(lL, x)
        
        # <lL|h(B1,AR;B2,AR)|rR>
        x = tm.eps_r_op_2s_A(rR, B1, AR, B2, AR, h_nn)
        res4 = mm.adot(lL, x)
        
        matels.append(res1 + res2 + res3 + res4)
    
    B1L = tm.eps_l_noop(lL, B1, AL) # <lL|(B1;AL)
    B2_KR = tm.eps_r_noop(KR, AR, B2) # (AR;B2)|KR>
    B2_KR += tm.eps_r_op_2s_A(rR, AR, AR, B2, AR, h_nn) # (AR;B2)|KR> + h(AR,AR;B2,AR)|rR>
    
    if d_max >= 1:
        res = mm.adot(B1L, B2_KR) # <lL|(B1;AL) (AR;B2)|KR> + <lL|(B1;AL) h(AR,AR;B2,AR)|rR>
        
        x = tm.eps_r_op_2s_A(rR, B1, AR, AL, B2, h_nn) # <lL|h(B1,AR;AL,B2)|rR>
        res += mm.adot(lL, x)
        
        matels.append(res)
    
    B2_KR = tm.eps_r_noop(B2_KR, AR, AL) # advance one site (d == 2)
    # (AR;AL)(AR;B2)|KR> + (AR;AL) h(AR,AR;B2,AR)|rR> + h(AR,AR;AL,B2)|rR>
    B2_KR += tm.eps_r_op_2s_A(rR, AR, AR, AL, B2, h_nn)
    
    for d in range(2, d_max + 1):
        x = B1L
        for _ in range(2, d):
            x = tm.eps_l_noop(x, AR, AL) # (AR;AL)^(d-2)
        res = mm.adot(x, B2_KR)
        matels.append(res)
    
    return matels

def check_kinematics(J, hx, hz, J2, J3, J4, J5, hzx, D):
  ham = get_ham(J, hx, 0.0, J2, J3, J4, J5, 0.0)
  try:
    state = tdvp.EvoMPS_TDVP_Uniform.from_file("vac_true_uniform.npy", ham)
    print("Loaded existing state from vac_true_uniform.npy")
  except FileNotFoundError:
    state = tdvp.EvoMPS_TDVP_Uniform(4, 2, ham, L=1)
  state = dyn.find_ground(
    state, tol=1e-11, h_init=0.01, cb_func=cbf, max_itr=1000, expand_to_D=D, expand_step=2)
  if state.expect_1s(Sz).real < 1:
    state.apply_op_1s(Sx)
  state.update()
  state.save_state("vac_symm_uniform.npy")

  state_flipped = copy.deepcopy(state)
  state_flipped.apply_op_1s(Sx)
  state_flipped.update()

  state_flipped.phase_align(state)

  print("Unbroken correlation length:", state.correlation_length())
  print("Smallest Schmidt-squared:", state.schmidt_sq().real.min())

  ens_triv = []
  ens_top = []
  ps = np.linspace(0.0, np.pi, num=40)
  for p in ps:
    ens_triv.append(np.sort(state.excite_top_triv_brute(p)))
    ens_top.append(np.sort(state.excite_top_nontriv_brute(state_flipped, p, phase_align=False)))
    print(p, ens_triv[-1].min(), ens_top[-1].min())
  
  ens_triv = np.array(ens_triv)
  ens_top = np.array(ens_top)
  np.save("symm_scalar_spec.npy", ens_triv)
  np.save("symm_scalar_spec_ps.npy", ps)
  np.save("symm_kink_spec.npy", ens_top)
  np.save("symm_kink_spec_ps.npy", ps)

  spec_info = {
    "gap_triv_p0": ens_triv[0][0],
    "gap_top_p0": ens_top[0][0],
  }
  with open("spec_info_symm.json", "w") as f:
    json.dump(spec_info, f)

  E_triv_0 = ens_triv[0].min()
  E_kink_0 = ens_top[0].min()

  es_kink_min = ens_top.T[0, :]
  ind = np.argmax(np.diff(es_kink_min))
  print("p(v_max) ~", ps[ind])
  E_kink_1 = ens_top[ind].min()

  print("E_triv(0) = ", E_triv_0)
  print("E_kink(0) = ", E_kink_0)
  print("E_kink(v_max) = ", E_kink_1)
  print("KE_kink(v_max) = ", E_kink_1 - E_kink_0)

  if 2*E_kink_0 > E_triv_0:
    print("Possible non-mesonic trivial excitation!")
  
  if E_kink_1 > E_triv_0:
    print("Kinks at v_max surpass scalar-production threshold!")

  E_kinetic = E_kink_1 - E_kink_0
  sep_required = E_kinetic / hz
  print("Required separation approx.:", sep_required)




def prepare_vacua(J, hx, hz, J2, J3, J4, J5, hzx, D, save_states=True):
  symm_broken = (hz != 0) or (hzx != 0)

  ham = get_ham(J, hx, hz, J2, J3, J4, J5, hzx)
  try:
    state = tdvp.EvoMPS_TDVP_Uniform.from_file("vac_true_uniform.npy", ham)
    print("Loaded existing state from vac_true_uniform.npy")
  except FileNotFoundError:
    state = tdvp.EvoMPS_TDVP_Uniform(4, 2, ham, L=1)
  
  state = dyn.find_ground(
    state, tol=1e-11, h_init=0.01, cb_func=cbf, max_itr=1000, expand_to_D=D, expand_step=2)
  state.update()

  h_orig = state.h_expect.real
  z_orig = state.expect_1s(Sz).real
  print("Correlation length:", state.correlation_length())
  print("Energy:", h_orig)
  print("<Z>:", z_orig)
  print("Schmidt coeff:", np.sqrt(state.schmidt_sq().real))

  np.save("state_info.npy", np.array([
    state.h_expect, state.expect_1s(Sz), state.correlation_length()]))

  try:
    state_flipped = tdvp.EvoMPS_TDVP_Uniform.from_file("vac_false_uniform.npy", ham)
    print("Loaded existing state from vac_false_uniform.npy")
  except FileNotFoundError:
    state_flipped = copy.deepcopy(state)
    state_flipped.apply_op_1s(Sx)
  
  state_flipped.update()

  h_flipped = state_flipped.h_expect.real
  z_flipped = state_flipped.expect_1s(Sz).real
  print("Flipped energy:", h_flipped)
  print("Flipped <Z>:", z_flipped)

  if symm_broken:
    # Flipped state is not an eigenstate: "cool it down"
    state_flipped = dyn.find_ground(
      state_flipped, tol=1e-11, h_init=0.01, cb_func=cbf, max_itr=1000, expand_to_D=D, expand_step=2)
    state_flipped.update()

    if state_flipped.eta.real > 1e-11:
      raise RuntimeError("Failed to cool flipped state.")

  h_flipped = state_flipped.h_expect.real
  z_flipped = state_flipped.expect_1s(Sz).real
  print("Flipped and cooled energy:", h_flipped)
  print("Flipped and cooled <Z>:", z_flipped)

  if symm_broken and np.sign(z_flipped) == np.sign(z_orig):
    raise RuntimeError("Failed to preserve metastable vacuum.")

  print("Flipped and cooled correlation length:", state_flipped.correlation_length())
  print("Schmidt coeff:", np.sqrt(state_flipped.schmidt_sq().real))
  print("Energy density gap:", h_flipped - h_orig)

  if symm_broken and (h_flipped < h_orig) or (not symm_broken) and (z_flipped > z_orig):
    vac_true, vac_false = state_flipped, state
  else:
    vac_false, vac_true = state_flipped, state

  if save_states:
    vac_true.save_state("vac_true_uniform.npy")
    vac_false.save_state("vac_false_uniform.npy")
  
  return vac_true, vac_false


def extrap_zero(s):
  # Assumes s is sorted
  if s[0] > 0.0:
    i = np.argmax(s < 0.0)
  else:
    i = np.argmax(s > 0.0)
  a, b = s[[i-1, i]]
  d = abs(a/(b - a))
  return i + d - 1


def prepare_state(J, hx, hz, J2, J3, J4, J5, hzx, D, N, pkt_numsites, pkt_sigma, pkt_midpt,
  pad_out=0, pad_mid=0, ortho_2p=False, symmetrize_B=False,
  truevac_outer=True, truevac_inner=False,
  momentum=0, save_file=True, kink_lvl=0, path=""):
  ham = get_ham(J, hx, hz, J2, J3, J4, J5, hzx)
  vac_true = tdvp.EvoMPS_TDVP_Uniform.from_file(path + "vac_true_uniform.npy", ham)
  vac_false = tdvp.EvoMPS_TDVP_Uniform.from_file(path + "vac_false_uniform.npy", ham)

  vac_out = vac_true if truevac_outer else vac_false
  vac_in = vac_true if truevac_inner else vac_false

  # Do this once so that the phase alignment holds for all excitations, and is
  # preserved in the sandwich state.
  if vac_out is not vac_in:
    vac_out.phase_align(vac_in)
  #vac_out.update(restore_CF=False)

  vac_out.save_state(path + "vac_out_uniform.npy")
  vac_in.save_state(path + "vac_in_uniform.npy")

  vac_out_L = copy.deepcopy(vac_out)
  vac_out_R = copy.deepcopy(vac_out)

  vac_in_T = copy.deepcopy(vac_in)
  vac_in_T.A[0] = vac_in_T.A[0].transpose((0,2,1))
  vac_in_T.update(restore_CF=False)

  force_pseudo = (truevac_inner == truevac_outer)
    
  if ortho_2p:
    vac_out_L_T = copy.deepcopy(vac_out_L)
    vac_out_L_T.A[0] = vac_out_L_T.A[0].transpose((0,2,1))
    vac_out_L_T.update(restore_CF=False)

    ens_kink, Bs_L_T = top_excitations(vac_in_T, vac_out_L_T, p=-momentum, force_pseudo=force_pseudo)
    print("Kink energy 1 (-inf):", ens_kink[kink_lvl])

    if momentum == 0.0:
      ev_pi = vac_in_T.excite_top_nontriv_brute(vac_out_L_T, np.pi, force_pseudo=force_pseudo)
      if ev_pi[kink_lvl] < ens_kink[kink_lvl]:
        print("Warning!: Kink at p=pi has a lower energy ({}) than at p=0.0 ({})!".format(
          ev_pi[kink_lvl], ens_kink[kink_lvl]
        ))

    ens2, Bs_R = top_excitations(vac_in, vac_out_R, p=-momentum, force_pseudo=force_pseudo)
    print("Kink energy 2 (-inf):", ens2[kink_lvl])
    if (ens_kink[kink_lvl] - ens2[kink_lvl]) > 1e-6:
      print("Warning!: Refl. kink energy {} does not match {}!".format(
        ens_kink[kink_lvl], ens2[kink_lvl]
      ))

    BL = Bs_L_T[kink_lvl].transpose((0,2,1))
    BR = Bs_R[kink_lvl]
  else:
    ens_kink, Bs_L = top_excitations(vac_out_L, vac_in, p=momentum, force_pseudo=force_pseudo)
    print("Kink energy 1 (-inf):", ens_kink[kink_lvl])

    if momentum == 0.0:
      ev_pi = vac_out_L.excite_top_nontriv_brute(vac_in, np.pi, force_pseudo=force_pseudo)
      if ev_pi[kink_lvl] < ens_kink[kink_lvl]:
        print("Warning!: Kink at p=pi has a lower energy ({}) than at p=0.0 ({})!".format(
          ev_pi[kink_lvl], ens_kink[kink_lvl]
        ))

    vac_out_R_T = copy.deepcopy(vac_out_R)
    vac_out_R_T.A[0] = vac_out_R_T.A[0].transpose((0,2,1))
    vac_out_R_T.update(restore_CF=False)

    ens2, Bs_R_T = top_excitations(vac_out_R_T, vac_in_T, p=momentum, force_pseudo=force_pseudo)
    print("Kink energy 2 (-inf):", ens2[kink_lvl])
    if (ens2[kink_lvl] - ens_kink[kink_lvl]) > 1e-6:
      print("Warning!: Refl. kink energy {} does not match {}!".format(
        ens2[kink_lvl], ens_kink[kink_lvl]
      ))

    BL = Bs_L[kink_lvl]
    BR = Bs_R_T[kink_lvl].transpose((0,2,1))

  if symmetrize_B:
    import bsymm
    force_orth_vac = force_pseudo and momentum != 0
    BL = bsymm.symmetrize_B(BL, vac_out_L, vac_in, p=momentum, force_orth_vac=force_orth_vac)
    BR = bsymm.symmetrize_B(BR, vac_in, vac_out_R, p=-momentum, force_orth_vac=force_orth_vac)

  pkt = gaussian_packet_coeff(pkt_numsites, pkt_sigma, pkt_midpt, p=momentum)
  print("Packet coeff. ends:", pkt[0], pkt[-1])

  pktL = np.array([0.0] * pad_out + list(pkt) + [0.0] * pad_mid)
  pktR = np.array([0.0] * pad_mid + list(pkt[::-1]) + [0.0] * pad_out)

  swp = scattering_state(N, vac_out_L, vac_out_R, vac_in,
                        BL, BR,
                        pktL, pktR,
                        zero_threshold=1e-12)
  swp.update()

  e_bulk = vac_out.h_expect.real
  if len(ham.shape) == 6:
    es_swp = np.array([swp.expect_3s(ham, j).real - e_bulk for j in range(swp.N + 1)])
  elif len(ham.shape) == 4:
    es_swp = np.array([swp.expect_2s(ham, j).real - e_bulk for j in range(swp.N + 1)])
  else:
    raise ValueError("Weird hamiltonian.")
  print("Energy:", sum(es_swp))

  if save_file:
    np.save(path + "initial_state_es.npy", es_swp)
    swp.save_state(path + "initial_state.npy")
    with open(path + "initial_state.pickle", "wb") as f:
      pickle.dump(swp, f)

  return swp, ens_kink


def check_vacs(J, hx, hz, J2, J3, J4, J5, hzx, D, truevac_outer=True, truevac_inner=False):
  ham = get_ham(J, hx, hz, J2, J3, J4, J5, hzx)
  vac_true = tdvp.EvoMPS_TDVP_Uniform.from_file("vac_true_uniform.npy", ham)
  vac_false = tdvp.EvoMPS_TDVP_Uniform.from_file("vac_false_uniform.npy", ham)

  vac_out = vac_true if truevac_outer else vac_false
  vac_in = vac_true if truevac_inner else vac_false

  print("Outer vacuum:")
  print("Correlation length:", vac_out.correlation_length())
  print("Energy:", vac_out.h_expect.real)
  print("<Z>:", vac_out.expect_1s(Sz).real)
  print("Schmidt coeff:", np.sqrt(vac_out.schmidt_sq().real))

  print("Inner vacuum:")
  print("Correlation length:", vac_in.correlation_length())
  print("Energy:", vac_in.h_expect.real)
  print("<Z>:", vac_in.expect_1s(Sz).real)
  print("Schmidt coeff:", np.sqrt(vac_in.schmidt_sq().real))
  
  vac_info = {
    "e_out": vac_out.h_expect.real,
    "e_in": vac_in.h_expect.real,
    "cl_out": vac_out.correlation_length(),
    "cl_in": vac_in.correlation_length(),
    "Z_out": vac_out.expect_1s(Sz).real,
    "Z_in": vac_in.expect_1s(Sz).real,
    "minschmidt_out": np.sqrt(vac_out.schmidt_sq().real.min()),
    "minschmidt_in": np.sqrt(vac_in.schmidt_sq().real.min())
  }
  with open("vac_info.json", "w") as f:
    json.dump(vac_info, f)


def check_kinematics_brk(J, hx, hz, J2, J3, J4, J5, hzx, D, N, pkt_numsites, pkt_sigma, pkt_midpt,
  pad_out=0, pad_mid=0, ortho_2p=False, symmetrize_B=False,
  truevac_outer=True, truevac_inner=False, momentum=0, p_vmax=1.0, kink_lvl=0):
  ham = get_ham(J, hx, hz, J2, J3, J4, J5, hzx)
  vac_true = tdvp.EvoMPS_TDVP_Uniform.from_file("vac_true_uniform.npy", ham)
  vac_false = tdvp.EvoMPS_TDVP_Uniform.from_file("vac_false_uniform.npy", ham)

  vac_out = vac_true if truevac_outer else vac_false
  vac_in = vac_true if truevac_inner else vac_false

  swp0, Es_kink0 = prepare_state(
    J, hx, hz, J2, J3, J4, J5, hzx, D, N, pkt_numsites, pkt_sigma, pkt_midpt,
    pad_out=pad_out, pad_mid=pad_mid, ortho_2p=ortho_2p, symmetrize_B=symmetrize_B,
    truevac_outer=truevac_outer, truevac_inner=truevac_inner,
    save_file=False, momentum=momentum, kink_lvl=kink_lvl)

  swp1, Es_kink1 = prepare_state(
    J, hx, hz, J2, J3, J4, J5, hzx, D, N, pkt_numsites, pkt_sigma, pkt_midpt,
    pad_out=pad_out, pad_mid=pad_mid, ortho_2p=ortho_2p, symmetrize_B=symmetrize_B,
    truevac_outer=truevac_outer, truevac_inner=truevac_inner,
    save_file=False, momentum=p_vmax, kink_lvl=kink_lvl)

  if vac_out.ham_sites == 2:
    expvals_en_0 = [swp0.expect_2s(vac_out.ham, i).real - vac_out.h_expect.real for i in range(N)]
    expvals_en_1 = [swp1.expect_2s(vac_out.ham, i).real - vac_out.h_expect.real for i in range(N)]
  elif vac_out.ham_sites == 3:
    expvals_en_0 = [swp0.expect_3s(vac_out.ham, i).real - vac_out.h_expect.real for i in range(N)]
    expvals_en_1 = [swp1.expect_3s(vac_out.ham, i).real - vac_out.h_expect.real for i in range(N)]

  e_gap = vac_in.h_expect.real - vac_out.h_expect.real
  print("e gap:", e_gap)

  E_bubble_0 = np.sum(expvals_en_0)
  print("Bubble energy (p={}):".format(momentum), E_bubble_0)
  E_bubble_1 = np.sum(expvals_en_1)
  print("Bubble energy (p={}):".format(p_vmax), E_bubble_1)

  expvals_swp_0 = np.array([swp0.expect_1s(Sz, i).real for i in range(N+2)])
  cross1 = extrap_zero(expvals_swp_0[:N//2])
  cross2 = extrap_zero(expvals_swp_0[N//2:]) + N//2
  sep_approx_0 = cross2 - cross1
  E_sep_0 = e_gap * sep_approx_0
  print("Pkt. crossings: ", cross1, cross2)
  print("Approx. seperation = ", sep_approx_0)
  print("Approx. sep. energy = ", E_sep_0)

  expvals_swp_1 = np.array([swp1.expect_1s(Sz, i).real for i in range(N+2)])
  cross1 = extrap_zero(expvals_swp_1[:N//2])
  cross2 = extrap_zero(expvals_swp_1[N//2:]) + N//2
  sep_approx_1 = cross2 - cross1
  E_sep_1 = e_gap * sep_approx_1
  print("Pkt. crossings: ", cross1, cross2)
  print("Approx. seperation = ", sep_approx_1)
  print("Approx. sep. energy = ", E_sep_1)

  ens_triv = vac_out.excite_top_triv_brute(0.0)
  E_triv_0 = ens_triv.real.min()
  print("E_triv(0) = ", E_triv_0)

  ens_triv = vac_out.excite_top_triv_brute(0.5)
  E_triv_half = ens_triv.real.min()
  print("E_triv(1/2) = ", E_triv_half)

  if E_bubble_0 > 2*E_triv_0:
    print("E_bubble > 2*E_triv(0): Scalar-pair threshold achieved!")
  if E_bubble_0 > 2*E_triv_half:
    print("E_bubble > 2*E_triv(1/2): Faster scalar-pair threshold achieved!")

  E_kink_0 = (E_bubble_0 - E_sep_0)/2
  E_kink_1 = (E_bubble_1 - E_sep_1)/2
  KE_bubbles = E_kink_1 - E_kink_0
  print("KE(vmax) from bubbles:", KE_bubbles)
  sep_p1_bubbles = 2 * KE_bubbles / e_gap
  print("vmax separation approx. (bubbles):", sep_p1_bubbles)

  KE_ex = Es_kink1.real.min() - Es_kink0.real.min()
  print("KE(vmax) from ex.:", KE_ex)
  sep_p1_ex = 2 * KE_ex / e_gap
  print("vmax separation approx. (ex):", sep_p1_ex)

  if 2*KE_bubbles < E_sep_0:
    print("Seperation below vmax threshold. Velocity probably below maximum.")

  print("Approx. kink energy (p={}) = ".format(momentum), E_kink_0)
  print("Approx. kink energy (p={}) = ".format(p_vmax), E_kink_1)

  sep_pkts = 2 * (pkt_numsites - pkt_midpt + pad_mid)
  E_sep_pkts = e_gap * sep_pkts
  print("Sep. packets = ", sep_pkts)
  print("Packet sep. energy = ", E_sep_pkts)

  if 2 * E_kink_0 > E_triv_0:
    print("Lighter scalar condition achieved!")


def save_specs(J, hx, hz, J2, J3, J4, J5, hzx, D, pmax=np.pi, truevac_outer=True, truevac_inner=False):
  ham = get_ham(J, hx, hz, J2, J3, J4, J5, hzx)
  vac_true = tdvp.EvoMPS_TDVP_Uniform.from_file("vac_true_uniform.npy", ham)
  vac_false = tdvp.EvoMPS_TDVP_Uniform.from_file("vac_false_uniform.npy", ham)

  vac_out = vac_true if truevac_outer else vac_false
  vac_in = vac_true if truevac_inner else vac_false

  if vac_out is not vac_in:
    vac_out.phase_align(vac_in)

  ens_triv = []
  ens_triv_false = []
  ens_top = []
  ps = np.linspace(0.0, pmax, num=40)
  for p in ps:
    ens_triv.append(np.sort(vac_out.excite_top_triv_brute(p)))
    if vac_in is not vac_out:
      ens_triv_false.append(np.sort(vac_in.excite_top_triv_brute(p)))
    ens_top.append(np.sort(vac_out.excite_top_nontriv_brute(vac_in, p, phase_align=False)))
    print(p, ens_triv[-1][0], ens_top[-1][0])
  
  ens_triv = np.array(ens_triv)
  ens_triv_false = np.array(ens_triv_false)
  ens_top = np.array(ens_top)
  np.save("vac_out_scalar_spec.npy", ens_triv)
  np.save("vac_out_scalar_spec_ps.npy", ps)
  np.save("vac_in_scalar_spec.npy", ens_triv_false)
  np.save("vac_in_scalar_spec_ps.npy", ps)
  np.save("vac_out_kink_spec.npy", ens_top)
  np.save("vac_out_kink_spec_ps.npy", ps)

  spec_info = {
    "gap_triv_p0": ens_triv[0][0],
  }
  with open("spec_info.json", "w") as f:
    json.dump(spec_info, f)


def check_specs(J, hx, hz, J2, J3, J4, J5, hzx, D, pmax=np.pi, truevac_outer=True, truevac_inner=False):
  ham = get_ham(J, hx, hz, J2, J3, J4, J5, hzx)
  vac_true = tdvp.EvoMPS_TDVP_Uniform.from_file("vac_true_uniform.npy", ham)
  vac_false = tdvp.EvoMPS_TDVP_Uniform.from_file("vac_false_uniform.npy", ham)
  vac_out = vac_true if truevac_outer else vac_false
  vac_in = vac_true if truevac_inner else vac_false
    
  gap_out = vac_out.excite_top_triv_brute(0.0)[0]
  gap_in = vac_in.excite_top_triv_brute(0.0)[0]

  print("estimated gap:", gap_out)
  print("inverse:", 1/gap_out)
    
  print("estimated gap (in):", gap_in)
  print("inverse (in):", 1/gap_in)

  print("cl:", vac_out.correlation_length().real)
  print("cl (in):", vac_in.correlation_length().real)
  
  spec_info = {
    "gap_triv_p0": gap_out,
    "gap_triv_p0_in": gap_in,
  }
  with open("spec_info.json", "w") as f:
    json.dump(spec_info, f)
    
  ham = get_ham(J, hx, 0.0, J2, J3, J4, J5, 0.0)
  vac_symm = tdvp.EvoMPS_TDVP_Uniform.from_file("vac_symm_uniform.npy", ham)
  vac_symm_2 = copy.deepcopy(vac_symm)
  vac_symm_2.apply_op_1s(Sx)
  vac_symm_2.update()
  vac_symm_2.phase_align(vac_symm)
    
  gap = vac_symm.excite_top_triv_brute(0.0)[0]
  gap_top = vac_symm.excite_top_nontriv_brute(vac_symm_2, 0.0)[0]
    
  spec_info = {
    "gap_triv_p0": gap,
    "gap_top_p0": gap_top,
  }
  with open("spec_info_symm.json", "w") as f:
    json.dump(spec_info, f)


def load_state(filename, ham_uni=None, do_update=True):
    s = tdvp_s.EvoMPS_MPS_Sandwich.from_file(filename)
    if ham_uni is not None:
        s.update(restore_CF=False)
        #s = tdvp_s.EvoMPS_TDVP_Sandwich.from_mps(s, ham, [ham] * (s.N + 1), ham)
        s.uni_l = tdvp.EvoMPS_TDVP_Uniform.from_mps(s.uni_l, ham_uni)
        s.uni_r = tdvp.EvoMPS_TDVP_Uniform.from_mps(s.uni_r, ham_uni)
    if do_update:
        s.update()
        if ham_uni is not None:
            s.uni_l.calc_C()
            s.uni_l.calc_K()
            s.uni_r.calc_C()
            s.uni_r.calc_K()
    return s


def evolve_split(Dmax, N, dt, t_end):
  def save_lists():
    np.save("expvals_evo.npy", np.vstack(expvals_evo))
    np.save("Ds_evo.npy", np.vstack(Ds_evo))
    np.save("truncerrs_lr.npy", truncerrs_lr)
    np.save("truncerrs_rl.npy", truncerrs_rl)
    np.save("schmidts_evo.npy", schmidts)

  # evo callback
  def cb_evo(s, i, truncerr_lr=None, truncerr_rl=None, **kwargs):
    # i == 0 means one step of dt was already done!
    t = t_start + (i + 1) * dt
    step_num = int(t / dt) - 1

    # split_step leaves us with correct r matrices, but not l.
    s.calc_l()

    if (step_num + 1) % int(1 / dt) == 0:
      scopy = copy.deepcopy(s)
      scopy.update()
      scopy.save_state("state_evo_step{}.npy".format(step_num))
      schmidt = [scopy.schmidt_sq(n) for n in range(scopy.N+2)]
    else:
      schmidt = None

    expvals_swp = np.array([s.expect_1s(Sz, n).real for n in range(s.N+2)])
    expvals_evo.append(expvals_swp)
    schmidts.append(schmidt)
    truncerrs_lr.append(truncerr_lr)
    truncerrs_rl.append(truncerr_rl)
    Ds_evo.append(copy.copy(swp.D))

    if (step_num + 1) % int(1 / dt) == 0:
      save_lists()
    print(
      step_num, t,
      np.argmin(np.abs(expvals_swp)),
      np.linalg.norm(np.nan_to_num(truncerr_lr)),
      np.linalg.norm(np.nan_to_num(truncerr_rl))
      )

  with open("initial_state.pickle", "rb") as f:
    swp = pickle.load(f)

  if os.path.isfile("expvals_evo.npy"):
    print("LOADING...")
    expvals_evo = list(np.load("expvals_evo.npy", allow_pickle=True))
    Ds_evo = list(np.load("Ds_evo.npy", allow_pickle=True))
    schmidts = list(np.load("schmidts_evo.npy", allow_pickle=True))
    truncerrs_lr = list(np.load("truncerrs_lr.npy", allow_pickle=True))
    truncerrs_rl = list(np.load("truncerrs_rl.npy", allow_pickle=True))

    print("Last step:", len(expvals_evo) - 1)
    stepnum_last_saved = len(expvals_evo) - 1 - (len(expvals_evo) % int(1 / dt))
    print("Last saved state at:", stepnum_last_saved)
  
    swp = load_state("state_evo_step{}.npy".format(stepnum_last_saved), ham_uni=swp.uni_l.ham)
    print(swp.D)

    test = np.array([swp.expect_1s(Sz, n).real for n in range(swp.N+2)])
    err = np.linalg.norm(test - expvals_evo[stepnum_last_saved])
    if err > 1e-6:
      raise ValueError("Loaded state does not match saved spin values!", err)

    t_start = (stepnum_last_saved + 1) * dt
    expvals_evo = expvals_evo[:stepnum_last_saved+1]
    Ds_evo = Ds_evo[:stepnum_last_saved+1]
    schmidts = schmidts[:stepnum_last_saved+1]
    truncerrs_lr = truncerrs_lr[:stepnum_last_saved+1]
    truncerrs_rl = truncerrs_rl[:stepnum_last_saved+1]
    print("LOADED!")
  else:
    if not swp.N == N:
      raise ValueError(
        "Loaded initial state size does not match {} != {}".format(swp.N, N))

    t_start = 0.0
    expvals_evo = []
    schmidts = []
    truncerrs_lr = []
    truncerrs_rl = []
    Ds_evo = []

  swp.N_centre = 1
  swp.update()

  swp.uni_r.calc_AA()
  swp.uni_r.calc_C()
  swp.uni_r.calc_K()
  swp.uni_l.calc_AA()
  swp.uni_l.calc_C()
  KL, _ = swp.uni_l.calc_K_l()
  ham_C = [swp.uni_r.ham]*(swp.N + 1)
  swp = ss.evolve_split(swp, ham_C, swp.uni_r.ham_sites,
    dt*1.j, int(t_end / dt),
    ncv=8, tol=dt**5,
    two_site=True, D_max=Dmax, min_schmidt=dt**5,
    switch_to_1site=True,
    KL_bulk=KL[-1], KR_bulk=swp.uni_r.K[0],
    cb_func=cb_evo,
    print_progress=True)


def evolve_RK4(Dmax, N, dt, t_end):
  def save_lists():
    np.save("expvals_evo.npy", np.vstack(expvals_evo))
    np.save("Ds_evo.npy", np.vstack(Ds_evo))
    np.save("eta_sqs_evo.npy", eta_sqs)
    np.save("schmidts_evo.npy", schmidts)
    np.save("proj_errs_evo.npy", proj_errs)

  # evo callback
  def cb_evo(s, i, **kwargs):
    # i == 0 *after* the first step of dt, in this case because we do not
    # run dyn.evolve() until after 1 initial step (evolve() calls the callback
    # once before doing any evolution.)
    t = t_start + (i + 1) * dt
    step_num = int(t / dt) - 1

    if (step_num + 1) % int(1 / dt) == 0:
      proj_err = s.compute_projection_error().real
      schmidt = [s.schmidt_sq(n) for n in range(s.N+2)]
      s.save_state("state_evo_step{}.npy".format(step_num))
    else:
      s.calc_l()
      proj_err = None
      schmidt = None

    eta = s.eta.real
    expvals_swp = np.array([s.expect_1s(Sz, n).real for n in range(s.N+2)])
    expvals_evo.append(expvals_swp)
    schmidts.append(schmidt)
    Ds_evo.append(copy.copy(swp.D))
    eta_sqs.append(s.eta_sq.real)
    proj_errs.append(proj_err)

    if (step_num + 1) % int(1 / dt) == 0:
      save_lists()
    print(
      step_num, t,
      np.argmin(np.abs(expvals_swp)),
      eta,
      s.maxD_is_less_than(Dmax),
      None if proj_err is None else np.sqrt(np.sum(proj_err))
      )

  with open("initial_state.pickle", "rb") as f:
    swp = pickle.load(f)

  if os.path.isfile("expvals_evo.npy"):
    print("LOADING...")
    expvals_evo = list(np.load("expvals_evo.npy", allow_pickle=True))
    Ds_evo = list(np.load("Ds_evo.npy", allow_pickle=True))
    eta_sqs = list(np.load("eta_sqs_evo.npy", allow_pickle=True))
    schmidts = list(np.load("schmidts_evo.npy", allow_pickle=True))
    proj_errs = list(np.load("proj_errs_evo.npy", allow_pickle=True))

    print("Last step:", len(expvals_evo) - 1)
    stepnum_last_saved = len(expvals_evo) - 1 - (len(expvals_evo) % int(1 / dt))
    print("Last saved state at:", stepnum_last_saved)

    swp = load_state("state_evo_step{}.npy".format(stepnum_last_saved), ham_uni=swp.uni_l.ham)

    test = np.array([swp.expect_1s(Sz, n).real for n in range(swp.N+2)])
    err = np.linalg.norm(test - expvals_evo[stepnum_last_saved])
    if err > 1e-6:
      raise ValueError("Loaded state does not match saved spin values!", err)

    t_start = (stepnum_last_saved + 1) * dt
    expvals_evo = expvals_evo[:stepnum_last_saved+1]
    Ds_evo = Ds_evo[:stepnum_last_saved+1]
    eta_sqs = eta_sqs[:stepnum_last_saved+1]
    schmidts = schmidts[:stepnum_last_saved+1]
    proj_errs = proj_errs[:stepnum_last_saved+1]
    print("LOADED!")
  else:
    if not swp.N == N:
      raise ValueError(
        "Loaded initial state size does not match {} != {}".format(swp.N, N))

    expvals_evo = []
    schmidts = []
    Ds_evo = []
    eta_sqs = []
    proj_errs = []

    t_start = 0.0

  # Do not assume we have a TDVP object here.
  swp = tdvp_s.EvoMPS_TDVP_Sandwich.from_mps(
    swp, swp.uni_l.ham, [swp.uni_l.ham]*(swp.N + 1), swp.uni_r.ham, ham_sites=swp.uni_l.ham_sites)

  # Initial split-step serves to increase the bond-dim in a well-conditioned way.
  maxD = np.max(swp.D)
  if t_start == 0.0 or maxD < Dmax:
    print("Performing initial split-step")
    print("Initial Ds=", [D for D in swp.D])
    swp.N_centre = 1
    swp.update()
    nsteps_init = 40  # be generous here, so that RK4 has a nicely-conditioned MPS to work with
    dt_init = dt / nsteps_init
    swp = ss.evolve_split(swp, swp.ham, swp.ham_sites, dt_init*1.j, nsteps_init,
        ncv=8, tol=dt_init**5,
        two_site=True, D_max=Dmax, min_schmidt=dt_init**5,
        switch_to_1site=True,
        KL_bulk=swp.K_l[0], KR_bulk=swp.K[swp.N + 1],
        print_progress=False)
    print("Final Ds=", [D for D in swp.D])
  else:
    # do a regular RK4 step without the callback
    dyn.evolve(swp, t_end, dt=dt,
      integ="rk4", dynexp=True, D_max=Dmax, sv_tol=dt**5, auto_truncate=False)
  swp.N_centre = N // 2

  # since `evolve()` calls the callback before the first step too, this will
  # record data for the state after the intial evolve_split/RK4 step above.
  dyn.evolve(swp, t_end, dt=dt, cb_func=cb_evo,
    integ="rk4", dynexp=True, D_max=Dmax, sv_tol=dt**5, auto_truncate=False)


if __name__ == "__main__":
  with open("state_params.json", "r") as f:
    state_params = json.load(f)
  if sys.argv[1] == "prep_vacs":
    prepare_vacua(
      state_params["J"],
      state_params["hx"],
      state_params["hz"],
      state_params["J2"],
      state_params["J3"],
      state_params["J4"],
      state_params["J5"],
      state_params["hzx"],
      state_params["D"])
  elif sys.argv[1] == "prep_pkts":
    prepare_state(
      state_params["J"],
      state_params["hx"],
      state_params["hz"],
      state_params["J2"],
      state_params["J3"],
      state_params["J4"],
      state_params["J5"],
      state_params["hzx"],
      state_params["D"],
      state_params["N"],
      state_params["pkt_numsites"],
      state_params["pkt_sigma"],
      state_params["pkt_midpt"],
      pad_mid=state_params["pad_mid"],
      pad_out=state_params["pad_out"],
      ortho_2p=state_params["ortho_2p"],
      symmetrize_B=state_params["symmB"],
      truevac_outer=state_params["truevac_outer"],
      truevac_inner=state_params["truevac_inner"],
      momentum=state_params["momentum"],
      kink_lvl=state_params["kink_lvl"])
  elif sys.argv[1] == "check":
    check_kinematics(
      state_params["J"],
      state_params["hx"],
      state_params["hz"],
      state_params["J2"],
      state_params["J3"],
      state_params["J4"],
      state_params["J5"],
      state_params["hzx"],
      state_params["D"])
  elif sys.argv[1] == "check_brk":
    check_kinematics_brk(
      state_params["J"],
      state_params["hx"],
      state_params["hz"],
      state_params["J2"],
      state_params["J3"],
      state_params["J4"],
      state_params["J5"],
      state_params["hzx"],
      state_params["D"],
      state_params["N"],
      state_params["pkt_numsites"],
      state_params["pkt_sigma"],
      state_params["pkt_midpt"],
      pad_mid=state_params["pad_mid"],
      pad_out=state_params["pad_out"],
      ortho_2p=state_params["ortho_2p"],
      symmetrize_B=state_params["symmB"],
      truevac_outer=state_params["truevac_outer"],
      truevac_inner=state_params["truevac_inner"],
      momentum=state_params["momentum"],
      p_vmax=float(sys.argv[2]),
      kink_lvl=state_params["kink_lvl"])
  elif sys.argv[1] == "save_spec":
    save_specs(
      state_params["J"],
      state_params["hx"],
      state_params["hz"],
      state_params["J2"],
      state_params["J3"],
      state_params["J4"],
      state_params["J5"],
      state_params["hzx"],
      state_params["D"],
      truevac_outer=state_params["truevac_outer"],
      truevac_inner=state_params["truevac_inner"])
  elif sys.argv[1] == "check_specs":
    check_specs(
      state_params["J"],
      state_params["hx"],
      state_params["hz"],
      state_params["J2"],
      state_params["J3"],
      state_params["J4"],
      state_params["J5"],
      state_params["hzx"],
      state_params["D"],
      truevac_outer=state_params["truevac_outer"],
      truevac_inner=state_params["truevac_inner"])
  elif sys.argv[1] == "check_vacs":
    check_vacs(
      state_params["J"],
      state_params["hx"],
      state_params["hz"],
      state_params["J2"],
      state_params["J3"],
      state_params["J4"],
      state_params["J5"],
      state_params["hzx"],
      state_params["D"],
      truevac_outer=state_params["truevac_outer"],
      truevac_inner=state_params["truevac_inner"])
  elif sys.argv[1] == "evolve":
    with open("evo_params.json", "r") as f:
      evo_params = json.load(f)

    if evo_params["integrator"] == "RK4":
      evolve_RK4(
        evo_params["Dmax"],
        state_params["N"],
        evo_params["dt"],
        evo_params["t_end"])
    elif evo_params["integrator"] == "split":
      evolve_split(
        evo_params["Dmax"],
        state_params["N"],
        evo_params["dt"],
        evo_params["t_end"])
    else:
      raise ValueError("invalid integrator: {}".format(evo_params["integrator"]))
  else:
    raise ValueError("invalid argument: '{}'".format(sys.argv[1]))
