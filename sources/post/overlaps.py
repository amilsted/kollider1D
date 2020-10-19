import copy
import pickle
import json
import numpy as np
import scipy.linalg as la
import numba
import evoMPS.tdvp_common as tm
import evoMPS.matmul as ma
import evoMPS.mps_sandwich as mps_s
import evoMPS.tdvp_uniform as tdvp_u

from multiprocessing.pool import ThreadPool

def top_excitations(sL, sR, p=0.0, phase_align=False, force_pseudo=False, brute=True, nev=10):
    if brute:
        ev, eV = sL.excite_top_nontriv_brute(
            sR, p,
            return_eigenvectors=True,
            phase_align=phase_align,
            force_pseudo=force_pseudo)
    else:
        ev, eV = sL.excite_top_nontriv(
            sR, p, nev=nev,
            return_eigenvectors=True,
            phase_align=phase_align,
            force_pseudo=force_pseudo)

    xs = [
    eV[:,i].reshape((sL.D, (sL.q - 1) * sL.D))
        for i in range(eV.shape[1])]

    Bs = [
        sL.get_B_from_x(
          x, sR.Vsh[0], sL.l_sqrt_i[0], sR.r_sqrt_i[0])
        for x in xs]

    return ev, Bs


def triv_excitations(s, p=0.0, brute=True, nev=10):
    if brute:
        ev, eV = s.excite_top_triv_brute(
            p,
            return_eigenvectors=True)
    else:
        ev, eV = s.excite_top_nontriv(
            p, nev=nev,
            return_eigenvectors=True)

    xs = [
    eV[:,i].reshape((s.D, (s.q - 1) * s.D))
        for i in range(eV.shape[1])]

    Bs = [
        s.get_B_from_x(
          x, s.Vsh[0], s.l_sqrt_i[0], s.r_sqrt_i[0])
        for x in xs]

    return ev, Bs


def excitation_tensors(vacL, falsevac, vacR, p=0.0, num_ex=1,
    verbose=True, force_pseudo=False, brute=True):
    falsevac = copy.deepcopy(falsevac)

    falsevac_T = copy.deepcopy(falsevac)
    falsevac_T.A[0] = falsevac_T.A[0].transpose((0,2,1))
    falsevac_T.update(restore_CF=False)

    vacL_T = copy.deepcopy(vacL)
    vacL_T.A[0] = vacL_T.A[0].transpose((0,2,1))
    vacL_T.update(restore_CF=False)

    #falsevac_T.phase_align(vacL_T)

    evs_L_T, Bs_L_T = top_excitations(falsevac_T, vacL_T, p=-p, force_pseudo=force_pseudo, brute=brute)
    if verbose:
        print("B1 spectrum:")
        print(evs_L_T)

    #falsevac.phase_align(vacR)

    evs_R, Bs_R = top_excitations(falsevac, vacR, p=-p, force_pseudo=force_pseudo, brute=brute)
    if verbose:
        print("B2 spectrum:")
        print(evs_R)
    
    B1s = [B.transpose((0,2,1)) for B in Bs_L_T[:num_ex]]
    B2s = Bs_R[:num_ex]

    return B1s, B2s


def gauge_match_2p(sw, stateL, B1, B2, stateR):
    if not (stateL.L == 1 and stateR.L == 1):
        raise ValueError("Bulk unit cell size must currently be 1.")

    stateL = copy.deepcopy(stateL)
    stateR = copy.deepcopy(stateR)
    B1 = B1.copy()
    B2 = B2.copy()
    
    # FIXME: uni_l is in LCF and may have small Schmidt coefficients.
    #        We probably want to perform gauge-alignment via uni_l.l rather
    #        than uni_l.r (which is what gauge_align currently does).
    _, gi, _ = stateL.gauge_align(sw.uni_l)
    for s in range(B1.shape[0]):
        B1[s,:,:] = gi[0].dot(B1[s,:,:])
    print("Tensor difference L: ", la.norm(stateL.A[0] - sw.uni_l.A[0]))

    g, _, _ = stateR.gauge_align(sw.uni_r)
    for s in range(B2.shape[0]):
        B2[s,:,:] = B2[s,:,:].dot(g[0])
    print("Tensor difference R: ", la.norm(stateR.A[0] - sw.uni_r.A[0]))
    
    return stateL, B1, B2, stateR


def basis_normalization_2p(stateL, BLs, stateC, BRs, stateR, max_d, verbose=True):
    if not (stateL.L == 1 and stateR.L == 1 and stateC.L == 1):
        raise ValueError("Bulk unit cell size must currently be 1.")

    for i, B in enumerate(BRs):
        check_gf = tm.eps_r_noop(stateR.r[0], B, stateR.A[0])
        if la.norm(check_gf) > 1e-10:
            print("Warning! BR{} RGF failed: {}".format(i, la.norm(check_gf)))
        elif verbose:
            print("BR{}, check RGF: {}".format(i, la.norm(check_gf)))

    for i, B in enumerate(BLs):
        check_gf = tm.eps_l_noop(stateL.l[0], B, stateL.A[0])
        if la.norm(check_gf) > 1e-10:
            print("Warning! BL{} LGF failed: {}".format(i, la.norm(check_gf)))
        elif verbose:
            print("BL{}, check LGF: {}".format(i, la.norm(check_gf)))

    ls = [tm.eps_l_noop(stateL.l[0], BL, BL) for BL in BLs]
    rs = [tm.eps_r_noop(stateR.r[0], BR, BR) for BR in BRs]
    ols = []
    for _ in range(max_d):
        ols.append([[ma.adot(l, r) for r in rs] for l in ls])
        ls = [tm.eps_l_noop(l, stateC.A[0], stateC.A[0]) for l in ls]

    return np.array(ols)


def basis_expvals_2p_1s_meh(op, stateL, BLs, stateC, BRs, stateR, max_d, verbose=True):
    if not (stateL.L == 1 and stateR.L == 1 and stateC.L == 1):
        raise ValueError("Bulk unit cell size must currently be 1.")

    for i, B in enumerate(BRs):
        check_gf = tm.eps_r_noop(stateR.r[0], B, stateR.A[0])
        if la.norm(check_gf) > 1e-10:
            print("Warning! BR{} RGF failed: {}".format(i, la.norm(check_gf)))
        elif verbose:
            print("BR{}, check RGF: {}".format(i, la.norm(check_gf)))

    for i, B in enumerate(BLs):
        check_gf = tm.eps_l_noop(stateL.l[0], B, stateL.A[0])
        if la.norm(check_gf) > 1e-10:
            print("Warning! BL{} LGF failed: {}".format(i, la.norm(check_gf)))
        elif verbose:
            print("BL{}, check LGF: {}".format(i, la.norm(check_gf)))

    l_op = tm.eps_l_op_1s(op, stateL.l[0], stateL.A[0], stateL.A[0])
    ls = [tm.eps_l_noop(stateL.l[0], BL, BL) for BL in BLs]
    rs = [tm.eps_r_noop(stateR.r[0], BR, BR) for BR in BRs]
    r_op = tm.eps_r_op_1s(op, stateR.r[0], stateR.A[0], stateR.A[0])
    r_op_BRs = [tm.eps_r_op_1s(op, stateR.r[0], BR, BR) for BR in BRs]
    
    # TODO: op on B sites!
    
    vals_left = np.zeros((max_d, max_d, len(BLs), len(BRs)), dtype=np.complex128)
    vals_mid = np.zeros((max_d, max_d, len(BLs), len(BRs)), dtype=np.complex128)
    vals_right = np.zeros((max_d, max_d, len(BLs), len(BRs)), dtype=np.complex128)
    for d1 in range(max_d):
        if d1 == 0:
            ls_BL_op = [tm.eps_l_op_1s(op, stateL.l[0], BL, BL) for BL in BLs]
        else:
            ls_BL_op = [tm.eps_l_op_1s(op, l, stateC.A[0], stateC.A[0]) for l in ls]

        ls_BRs = [[tm.eps_l_noop(l, BR, BR) for BR in BRs] for l in ls]
        for d2 in range(max_d):
            if d2 == 0:
                vals_right[d1, d2, :, :] = [[ma.adot(l, r) for r in r_op_BRs] for l in ls]
            else:
                vals_mid[d1, d2, :, :] = [[ma.adot(l, r) for r in rs] for l in ls_BL_op]
                vals_right[d1, d2, :, :] = [[ma.adot(l, r) for r in r_op] for l in ls_BRs]
                ls_BL_op = [tm.eps_l_noop(l, stateC.A[0], stateC.A[0]) for l in ls_BL_op]
                ls_BRs = [tm.eps_l_noop(l, stateR.A[0], stateR.A[0]) for l in ls_BRs]
        
        if d1 > 0:
            ls = [tm.eps_l_noop(l, stateC.A[0], stateC.A[0]) for l in ls]
        
        
    

    return np.array(ols)


def basis_expvals_2p_mpo_right(mpo, stateL, BL, BLc, stateC, stateCc, BR, BRc, stateR, N, verbose=True):
    """Expectation values in case MPO begins on same site as BR, or to the right of it.
    
    Returns vals[d1,d2].
    d1 is distance from BL to BR. This is only defined > 0.
    d2 is distance BR to first site of MPO. Define >= 0.
    """
    if not (stateL.L == 1 and stateR.L == 1 and stateC.L == 1):
        raise ValueError("Bulk unit cell size must currently be 1.")
    
    M = len(mpo)
    AL = stateL.A[0]
    AC = stateC.A[0]
    ACc = stateCc.A[0]
    AR = stateR.A[0]
    r = stateR.r[0]
    
    vals = np.full((N, N), np.NaN, dtype=np.complex128)
    
    # precompute the mpo transfer vector from the right
    r_op = np.expand_dims(r, axis=0)
    for l in reversed(range(M)):
        if l == 0:
            r_op_BR = tm.eps_r_op_MPO(r_op, BR, BRc, mpo[l])
        r_op = tm.eps_r_op_MPO(r_op, AR, AR, mpo[l])
    r_op = np.squeeze(r_op, axis=0)
    r_op_BR = np.squeeze(r_op_BR, axis=0)
    
    lBL = tm.eps_l_noop(stateL.l[0], BL, BLc)
    for d1 in range(1, N):
        vals[d1, 0] = np.vdot(lBL, r_op_BR)
        lBLBR = tm.eps_l_noop(lBL, BR, BRc)
        
        # This is less efficient than it needs to be. We could precompute
        # the TM ops in this loop (from the right).
        for d2 in range(1, N - d1):
            vals[d1, d2] = np.vdot(lBLBR, r_op)
            lBLBR = tm.eps_l_noop(lBLBR, AR, AR)

        lBL = tm.eps_l_noop(lBL, AC, ACc)

    return vals


def basis_expvals_2p_mpo_right_offdiag(mpo, stateL, BL, BLc, stateC, stateCc, BR, BRc, stateR, N, verbose=True):
    """Expectation values.
    """
    if not (stateL.L == 1 and stateR.L == 1 and stateC.L == 1):
        raise ValueError("Bulk unit cell size must currently be 1.")
    
    M = len(mpo)
    AL = stateL.A[0]
    AC = stateC.A[0]
    ACc = stateCc.A[0]
    AR = stateR.A[0]
    r = stateR.r[0]
    
    # precompute the mpo transfer vectors from the right
    r_op = np.expand_dims(r, axis=0)
    
    # mpo TM including both a BR tensor
    r_op_BRs = {}
    # mpo TM including both BR tensors, BR at same site, or to right of BRc
    r_op_BRBRs = {}
    for l in reversed(range(M)):  # NOTE: Squeezing not needed for vdots
        r_op_BRs[l] = tm.eps_r_op_MPO(r_op, BR, AR, mpo[l])
        r_op_BRBRs[l, l] = tm.eps_r_op_MPO(r_op, BR, BRc, mpo[l])
        x = tm.eps_r_op_MPO(r_op, BR, AR, mpo[l])
        for lc in reversed(range(l)):
            r_op_BRBRs[lc, l] = tm.eps_r_op_MPO(x, AC, BRc, mpo[lc])
            if lc > 0:
                x = tm.eps_r_op_MPO(x, AC, AR, mpo[lc])
        r_op = tm.eps_r_op_MPO(r_op, AR, AR, mpo[l])
    r_op = np.squeeze(r_op, axis=0)

    # precompute op TM vector from right until convergence
    # NOTE: This depends on stateR having consistent l and r matrices.
    r_ops = [r_op]
    for i in range(N-1):
        r_op = tm.eps_r_noop(r_ops[-1], AR, AR)
        diff = la.norm(r_op - np.vdot(np.asarray(stateR.l[0]), r_op)*r)
        r_ops.append(r_op)
        if diff < 1e-12:
            # NOTE: If this happens, terms involving r_ops[-1] are zero.
            print("L stopping at", i, "with", diff)
            break
    op_d3_max = len(r_ops)
    if op_d3_max == N:
        print("WARNING: Did not converge for d3_max!")
    
    # Order: BLBLc<-d1->op<-d2->BRc<-d3->BR
    # BR and BRc both overlapping op.
    # Defined for d1 > 0, d2 >= 0, d3 >= 0
    vals_op_BRc = np.full((N, M, M), np.NaN, dtype=np.complex128)

    # Order: BLBLc<-d1->BRc<-d2->op<-d3->BR
    # BR and op overlapping.
    # Defined for d1 > 0, d2 > 0, d3 >= 0
    vals_BRc_op = np.full((N, N, M), np.NaN, dtype=np.complex128)

    # Order: BLBLc<-d1->BRc<-d2->BR<-d3->op
    # BRc, BR, and op all separated.
    # Defined for d1 > 0, d2 > 0, d3 > 0
    vals_BRc_BR_op = np.full((N, N, op_d3_max+1), np.NaN, dtype=np.complex128)
    
    lBL = tm.eps_l_noop(stateL.l[0], BL, BLc)
    conv_lBl = False
    d2_max = 0
    for d1 in range(1, N):
        # We are including the d2=0,d3=0 term here. This is also covered by basis_expvals_2p_mpo_right()
        #vals_BRc_op[d1, 0, :] = np.array([np.vdot(lBL, r_op_BRBRs[0, i]) for i in range(M)])
        
        # We are including l==lc terms, which are already covered by basis_expvals_2p_mpo_right()
        for lc in range(M):
            for l in range(lc, M):
                vals_op_BRc[d1, lc, l] = np.vdot(lBL, r_op_BRBRs[lc, l])
        
        if not conv_lBl:
            lBLBRc_init = tm.eps_l_noop(lBL, AC, BRc)
        
        lBLBRc = lBLBRc_init
        for d2 in range(1, N - d1):
            vals_BRc_op[d1, d2, :] = [np.vdot(lBLBRc, r_op_BRs[i]) for i in range(M)]

            lBLBRcBR = tm.eps_l_noop(lBLBRc, BR, AR)
            vals_BRc_BR_op[d1,d2,1:] = [np.vdot(lBLBRcBR, x) for x in r_ops]

            if d2 > 5:  # These ought to go to zero eventually
                diff1 = la.norm(vals_BRc_op[d1, d2-5:d2+1, :])
                diff2 = la.norm(vals_BRc_BR_op[d1, d2-5:d2+1, 1:])
                if diff1 < 1e-12 and diff2 < 1e-12:
                    # results will no longer change appreciably, so broadcast
                    # present d2 to all larger d2, then stop
                    d2_max = max(d2_max, d2)
                    vals_BRc_op[d1, d2:, :] = vals_BRc_op[d1, d2, :]
                    vals_BRc_BR_op[d1, d2:, :] = vals_BRc_BR_op[d1, d2, :]
                    break
            lBLBRc = tm.eps_l_noop(lBLBRc, AC, AR)
        
        if not conv_lBl:
            lBL_new = tm.eps_l_noop(lBL, AC, ACc)
            if la.norm(lBL_new - lBL) < 1e-12:
                print("lBL converged at", d1)
                conv_lBl = True
            lBL = lBL_new

    print("d2_max =", d2_max)
    vals_BRc_op = vals_BRc_op[:, :d2_max+1, :].copy()
    vals_BRc_BR_op = vals_BRc_BR_op[:, :d2_max+1, :].copy()

    return vals_op_BRc, vals_BRc_op, vals_BRc_BR_op


def basis_expvals_2p_mpo_mid(mpo, stateL, BL, BLc, stateC, stateCc, BR, BRc, stateR, N, verbose=True):
    """Expectation values in case MPO begins on same site as BL, or between BL and BR.
    
    Returns vals[d1,d2].
    d1 is distance from BL to the first site of the MPO. Only defined > 0.
    d2 is distance from first site of MPO to BR. Only defined > 0.
    """
    if not (stateL.L == 1 and stateR.L == 1 and stateC.L == 1):
        raise ValueError("Bulk unit cell size must currently be 1.")
    
    M = len(mpo)
    AL = stateL.A[0]
    AC = stateC.A[0]
    ACc = stateCc.A[0]
    AR = stateR.A[0]
    l = stateL.l[0]
    r = stateR.r[0]
    r_BR = tm.eps_r_noop(r, BR, BRc)
    
    # precompute the mpo transfer vectors from the right
    r_op = np.expand_dims(r, axis=0)
    r_op_BRs = {}
    for l in reversed(range(1,M)):
        r_op_BRs[l] = tm.eps_r_op_MPO(r_op, BR, BRc, mpo[l])
        if l > 1:
            r_op = tm.eps_r_op_MPO(r_op, AR, AR, mpo[l])
    
    vals = np.full((N, N), np.NaN, dtype=np.complex128)
    
    lBL = tm.eps_l_noop(l, BL, BLc)
    for d1 in range(1, N-1):
        l_BLop = tm.eps_l_op_MPO(np.expand_dims(lBL, axis=0), AC, ACc, mpo[0])
        if len(mpo) == 1:
            l_BLop = np.squeeze(l_BLop, axis=0)
        lBL = tm.eps_l_noop(lBL, stateC.A[0], stateC.A[0])

        # This is less efficient than it needs to be. We could precompute
        # the TM ops in this loop (from the right).
        for d2 in range(1, N-d1):
            if d2 < M:
                vals[d1,d2] = np.vdot(l_BLop, r_op_BRs[d2])
                l_BLop = tm.eps_l_op_MPO(l_BLop, AC, ACc, mpo[d2])
                if d2 == M-1:
                    l_BLop = np.squeeze(l_BLop, axis=0)
            else:
                vals[d1,d2] = np.vdot(l_BLop, r_BR)
                l_BLop = tm.eps_l_noop(l_BLop, AC, ACc)

    return vals


def basis_expvals_2p_mpo_left(mpo, stateL, BL, BLc, stateC, stateCc, BR, BRc, stateR, N, verbose=True):
    """Expectation values in case MPO begins to the left of, or on the same it as BL.
    
    Returns vals[d1,d2].
    d1 is distance from the first site of MPO to BL. Defined >= 0.
    d2 is distance from BL to BR. Only defined > 0.
    """
    if not (stateL.L == 1 and stateR.L == 1 and stateC.L == 1):
        raise ValueError("Bulk unit cell size must currently be 1.")
    
    M = len(mpo)
    AL = stateL.A[0]
    AC = stateC.A[0]
    ACc = stateCc.A[0]
    AR = stateR.A[0]
    l = stateL.l[0]
    r = stateR.r[0]
    r_BR = tm.eps_r_noop(stateR.r[0], BR, BRc)
    
    # precompute the mpo transfer vectors from the right
    r_op = np.expand_dims(r, axis=0)
    r_op_BRs = {}
    for l in reversed(range(1,M)):
        r_op_BRs[l] = tm.eps_r_op_MPO(r_op, BR, BRc, mpo[l])
        if l > 1:
            r_op = tm.eps_r_op_MPO(r_op, AR, AR, mpo[l])
    
    vals = np.full((N, N), np.NaN, dtype=np.complex128)
    
    l_op = np.expand_dims(l, axis=0)
    for d1 in range(N):
        if d1 < M:
            l_opBL = tm.eps_l_op_MPO(l_op, BL, BLc, mpo[d1])
            l_op = tm.eps_l_op_MPO(l_op, AL, AL, mpo[d1])
            if d1 == M - 1:
                l_opBL = np.squeeze(l_opBL, axis=0)
                l_op = np.squeeze(l_op, axis=0)
        else:
            l_opBL = tm.eps_l_noop(l_op, BL, BLc)
            l_op = tm.eps_l_noop(l_op, AL, AL)

        # This is less efficient than it needs to be. We could precompute
        # the TM ops in this loop (from the right).
        for d2 in range(1, N - d1):
            if d1 + d2 < M:
                vals[d1,d2] = np.vdot(l_opBL, r_op_BRs[d1+d2])
                l_opBL = tm.eps_l_op_MPO(l_opBL, AC, ACc, mpo[d1+d2])
                if d1 + d2 == M-1:
                    l_opBL = np.squeeze(l_opBL, axis=0)
            else:
                vals[d1,d2] = np.vdot(l_opBL, r_BR)
                l_opBL = tm.eps_l_noop(l_opBL, AC, ACc)

    return vals


def basis_expvals_2p_mpo_get_data(mpo, stateL, BL, BLc, stateC, stateCc, BR, BRc, stateR, N, verbose=True):
    nrmsqs = basis_normalization_2p(stateL, [BL], stateC, [BR], stateR, N+1)
    nrmsqs_c = basis_normalization_2p(stateL, [BLc], stateCc, [BRc], stateR, N+1)
    
    if verbose: print("Starting left...")
    left = basis_expvals_2p_mpo_left(mpo, stateL, BL, BLc, stateC, stateCc, BR, BRc, stateR, N, verbose)
    if verbose: print("Left done. Starting mid...")
    mid = basis_expvals_2p_mpo_mid(mpo, stateL, BL, BLc, stateC, stateCc, BR, BRc, stateR, N, verbose)
    if verbose: print("Mid done. Starting right...")
    right = basis_expvals_2p_mpo_right(mpo, stateL, BL, BLc, stateC, stateCc, BR, BRc, stateR, N, verbose)
    if verbose: print("Right done. Starting right, off-diag...")
    od_right1, od_right2, od_right3 = basis_expvals_2p_mpo_right_offdiag(mpo, stateL, BL, BLc, stateC, stateCc, BR, BRc, stateR, N, verbose)
    if verbose: print("Done. Normalising...")
    
    for d2 in range(1, left.shape[1]):
        left[:,d2] /= nrmsqs[d2-1, 0, 0]**0.5 * nrmsqs_c[d2-1, 0, 0]**0.5

    for d1 in range(1, mid.shape[0]):
        for d2 in range(1, mid.shape[1]-d1):
            mid[d1,d2] /= nrmsqs[d1+d2-1, 0, 0]**0.5 * nrmsqs_c[d1+d2-1, 0, 0]**0.5
            
    for d1 in range(1, right.shape[0]):
        right[d1,:] /= nrmsqs[d1-1, 0, 0]**0.5 * nrmsqs_c[d1-1, 0, 0]**0.5
    
    for d1 in range(od_right1.shape[0]):
        for d2 in range(od_right1.shape[1]):
            for d3 in range(od_right1.shape[2]):
                od_right1[d1,d2,d3] /= nrmsqs[d1+d2+d3-1, 0, 0]**0.5 * nrmsqs_c[d1+d2-1, 0, 0]**0.5
                
    for d1 in range(od_right2.shape[0]):
        for d2 in range(min(od_right2.shape[1], nrmsqs.shape[0]-d1)):
            for M in range(od_right2.shape[2]):
                od_right2[d1,d2,M] /= nrmsqs[d1+d2+M-1, 0, 0]**0.5 * nrmsqs_c[d1-1, 0, 0]**0.5

    for d1 in range(od_right3.shape[0]):
        for d2 in range(min(od_right3.shape[1], nrmsqs.shape[0]-d1)):
            od_right3[d1,d2,:] /= nrmsqs[d1+d2-1, 0, 0]**0.5 * nrmsqs_c[d1-1, 0, 0]**0.5
    
    return left, mid, right, od_right1, od_right2, od_right3


@numba.jit(nopython=True)
def basis_expvals_2p_mpo_diag(out, psi, psi_c, left, mid, right):
    N = psi.shape[0]
    for j1 in range(N-1):
        for j2 in range(j1+1, N):
            coeff = np.conj(psi_c[j1, j2]) * psi[j1,j2]
            for n in range(j1+1):
                out[n] += coeff * left[j1-n, j2-j1]
            for n in range(j1+1, j2):
                out[n] += coeff * mid[n-j1, j2-n]
            for n in range(j2, N):
                out[n] += coeff * right[j2-j1, n-j2]
    return out


@numba.jit(nopython=True)
def basis_expvals_2p_mpo_offdiag(out, psi, psi_c, right_od_BRc_op, right_od_BRc_BR_op):
    N = psi.shape[0]
    d2_len1 = right_od_BRc_op.shape[1]
    M = right_od_BRc_op.shape[2]
    d3_len2 = right_od_BRc_BR_op.shape[2]
    d2_len2 = right_od_BRc_BR_op.shape[1]
    for j1 in range(N-1):
        for j2 in range(j1+1, N):
            for n in range(j2, N):
                if n-j2 < d2_len1:
                    for j3 in range(max(n, j2+1), n+M):
                        #if j3 == j2: continue
                        out[n] += np.conj(psi_c[j1,j2]) * psi[j1,j3] * right_od_BRc_op[j2-j1, n-j2, j3-n]
                        out[n] += np.conj(psi_c[j1,j3]) * psi[j1,j2] * np.conj(right_od_BRc_op[j2-j1, n-j2, j3-n])
                for j3 in range(max(j2+1, n-d3_len2+1), min(n, j2+d2_len2)):
                    #if n-j3 >= d3_len2: continue
                    #if j3-j2 >= d2_len2: continue
                    out[n] += np.conj(psi_c[j1,j2]) * psi[j1,j3] * right_od_BRc_BR_op[j2-j1, j3-j2, n-j3]
                    out[n] += np.conj(psi_c[j1,j3]) * psi[j1,j2] * np.conj(right_od_BRc_BR_op[j2-j1, j3-j2, n-j3])
    return out


def basis_expvals_2p_mpo(psi, psi_c, left, mid, right, od_right1, od_right2, od_right3):
    expvals = np.zeros(psi.shape[0]-1, dtype=np.complex128)
    expvals = basis_expvals_2p_mpo_diag(expvals, psi[1:,1:], psi_c[1:,1:], left, mid, right)
    expvals = basis_expvals_2p_mpo_offdiag(expvals, psi[1:,1:], psi_c[1:,1:], od_right2, od_right3)

    # trick to get od_left terms! assumes reflection symmetry, which should be a good approximation
    expvals_flip = np.zeros(psi.shape[0]-1, dtype=np.complex128)
    expvals_flip = basis_expvals_2p_mpo_offdiag(expvals_flip, psi[1:,1:][::-1,::-1].T, psi_c[1:,1:][::-1,::-1].T, od_right2, od_right3)
    expvals += expvals_flip[::-1]
    return expvals


def prep_2p_comps(sL, state_middle, sR, ps=[0.0], num_ex=1,
    cross_terms=False, ortho=True,
    verbose=True, force_pseudo=False, brute=True):
    sL.phase_align(state_middle)
    sR.phase_align(state_middle)

    # if sw is not a TDVP, update() misses the following:
    sL.calc_C()
    sL.calc_K()
    sR.calc_C()
    sR.calc_K()

    B1s, B2s, ex_labels = _prep_2p_Bs(
        sL, state_middle, sR,
        ps=ps, num_ex=num_ex,
        cross_terms=cross_terms, ortho=ortho,
        verbose=verbose, force_pseudo=force_pseudo, brute=brute)
    
    return sL, sR, B1s, B2s, ex_labels


def basis_overlaps_2p_(stateL, BL1s, stateC1, BR1s, BL2s, stateC2, BR2s, stateR, max_d, verbose=True):
    if not (stateL.L == 1 and stateR.L == 1 and stateC1.L == 1 and stateC2.L == 1):
        raise ValueError("Bulk unit cell size must currently be 1.")

    for i, B in enumerate([*BR1s, *BR2s]):
        check_gf = tm.eps_r_noop(stateR.r[0], B, stateR.A[0])
        if la.norm(check_gf) > 1e-10:
            print("Warning! BR{} RGF failed: {}".format(i, la.norm(check_gf)))
        elif verbose:
            print("BR{}, check RGF: {}".format(i, la.norm(check_gf)))

    for i, B in enumerate([*BL1s, *BL2s]):
        check_gf = tm.eps_l_noop(stateL.l[0], B, stateL.A[0])
        if la.norm(check_gf) > 1e-10:
            print("Warning! BL{} LGF failed: {}".format(i, la.norm(check_gf)))
        elif verbose:
            print("BL{}, check LGF: {}".format(i, la.norm(check_gf)))

    ls = [tm.eps_l_noop(stateL.l[0], BL1, BL2) for BL1 in BL1s for BL2 in BL2s]
    rs = [tm.eps_r_noop(stateR.r[0], BR1, BR2) for BR1 in BR1s for BR2 in BR2s]
    ols = []
    for _ in range(max_d):
        ols.append([[ma.adot(l, r) for r in rs] for l in ls])
        ls = [tm.eps_l_noop(l, stateC1.A[0], stateC2.A[0]) for l in ls]

    ols = np.array(ols).reshape((max_d, len(BL1s), len(BL2s), len(BR1s), len(BR2s)))
    ols = ols.transpose((0, 1, 3, 2, 4))
    return ols


def overlap_2p_components_(sw, stateL, B1s, stateC, B2s, stateR, cross_terms=True, verbose=True):
    """Projects onto the (separated) 2-particle position basis.

    The particles are specified via the tensors B1 and B2. Basis states
    take the form:
    ```
    |psi_jk> = 
      ... - AL - AL - B1 - AC - ... - AC - B2 - AR - AR - ...
      ...   |    |    |    |    ...   |    |    |    |  - ...
                      j                    k
    ```
    where AL = stateL.A[0], AC = stateC.A[0], and AR = stateR.A[0]. These
    particles are assumed to be kinks, so k > j always.
    It is assumed that B1 is left gauge-fixing wrt. AL and B2 is right
    gauge-fixing wrt. AR, so that <psi_jk|psi_lm> = 0 unless j == l and k == m.
    """
    if not len(B1s) == len(B2s):
        raise ValueError("B1s and B2s have different lengths!")

    diffL = la.norm(stateL.A[0] - sw.uni_l.A[0])
    diffR = la.norm(stateR.A[0] - sw.uni_r.A[0])
    if diffL > 1e-10 or diffR > 1e-10:
        print("Warning! Bulk tensor differences: {}, {}".format(diffL, diffR))
    
    lL = stateL.l[0]
    rR = stateR.r[0]
       
    if not (stateL.L == 1 and stateR.L == 1 and stateC.L == 1):
        raise ValueError("Bulk unit cell size must currently be 1.")

    AR = stateR.A[0]
    for i, B2 in enumerate(B2s):
        check_gf = tm.eps_r_noop(stateR.r[0], B2, AR)
        if la.norm(check_gf) > 1e-10:
            print("Warning! B2s[{}] RGF failed:".format(i), la.norm(check_gf))
        elif verbose:
            print("B2s[{}], check RGF:".format(i), la.norm(check_gf))

    AL = stateL.A[0]
    for i, B1 in enumerate(B1s):
        check_gf = tm.eps_l_noop(stateL.l[0], B1, AL)
        if la.norm(check_gf) > 1e-10:
            print("Warning! B1s[{}] LGF failed:".format(i), la.norm(check_gf))
        elif verbose:
            print("B1s[{}], check LGF:".format(i), la.norm(check_gf))
    
    AC = stateC.A[0]
    
    N = sw.N
    
    ls = [lL, lL]
    for j in range(1, sw.N + 1):
        ls.append(tm.eps_l_noop(ls[-1], sw.A[j], AL))
    # ls[1] is the left half including site 0, so the l matrix needed at site 1.
    # ls[2] is the left half including site 1, so the l matrix needed at site 2.
    # etc...
    
    rs = [rR]
    for j in range(sw.N, 0, -1):
        rs.insert(0, tm.eps_r_noop(rs[0], sw.A[j], AR))
    # rs[0] is now the right half including site 1, so the r matrix needed
    # for computations involving site 0.

    rsB2s = [None]
    rsB2s += [
        [tm.eps_r_noop(rs[j2], sw.A[j2], B2) for B2 in B2s]
        for j2 in range(1, len(rs))]
    rs = None

    num_ex = len(B1s)
    if cross_terms:
        ols = np.zeros((N+1, N+1, num_ex, num_ex), dtype=np.complex128)
    else:
        ols = np.zeros((N+1, N+1, num_ex), dtype=np.complex128)
    ols.fill(np.NaN)
    for j1 in range(1, N):
        l_mids = [tm.eps_l_noop(ls[j1], sw.A[j1], B1) for B1 in B1s]
        ls[j1] = None
        for j2 in range(j1 + 1, N + 1):
            rB2s = rsB2s[j2]
            if cross_terms:
                ols[j1, j2, :, :] = [[ma.adot(l, rB2) for rB2 in rB2s] for l in l_mids]
            else:
                ols[j1, j2, :] = [ma.adot(l, rB2) for l, rB2 in zip(l_mids, rB2s)]
            l_mids = [tm.eps_l_noop(l, sw.A[j2], AC) for l in l_mids]

    return ols


def overlap_3p_components_(sw, stateL, BLs, stateC, BCs, BRs, stateR, verbose=True):
    """Projects onto the (separated) 2k+1p position basis.

    The particles are specified via the tensors B1 and B2. Basis states
    take the form:
    ```
    |psi_jkl> = 
      ... - AL - AL - BL - AC - ... - BC - ... - AC - BR - AR - AR - ...
      ...   |    |    |    |    ...   |    ...   |    |  - ...
                      j               k               l
    ```
    where AL = stateL.A[0], AC = stateC.A[0], and AR = stateR.A[0]. These
    particles are assumed to be kinks, so k > j always.
    It is assumed that B1 is left gauge-fixing wrt. AL and B2 is right
    gauge-fixing wrt. AR, so that <psi_jk|psi_lm> = 0 unless j == l and k == m.
    """
    if not len(BLs) == len(BRs):
        raise ValueError("BLs and BRs have different lengths!")
    if not len(BLs) == len(BCs):
        raise ValueError("BLs and BCs have different lengths!")

    diffL = la.norm(stateL.A[0] - sw.uni_l.A[0])
    diffR = la.norm(stateR.A[0] - sw.uni_r.A[0])
    if diffL > 1e-10 or diffR > 1e-10:
        print("Warning! Bulk tensor differences: {}, {}".format(diffL, diffR))
    
    lL = stateL.l[0]
    rR = stateR.r[0]
       
    if not (stateL.L == 1 and stateR.L == 1 and stateC.L == 1):
        raise ValueError("Bulk unit cell size must currently be 1.")

    AR = stateR.A[0]
    for i, BR in enumerate(BRs):
        check_gf = tm.eps_r_noop(stateR.r[0], BR, AR)
        if la.norm(check_gf) > 1e-10:
            print("Warning! BRs[{}] RGF failed:".format(i), la.norm(check_gf))
        elif verbose:
            print("BRs[{}], check RGF:".format(i), la.norm(check_gf))

    AL = stateL.A[0]
    for i, BL in enumerate(BLs):
        check_gf = tm.eps_l_noop(stateL.l[0], BL, AL)
        if la.norm(check_gf) > 1e-10:
            print("Warning! BLs[{}] LGF failed:".format(i), la.norm(check_gf))
        elif verbose:
            print("BLs[{}], check LGF:".format(i), la.norm(check_gf))
    
    AC = stateC.A[0]
    
    N = sw.N
    
    ls = [lL, lL]
    for j in range(1, sw.N + 1):
        ls.append(tm.eps_l_noop(ls[-1], sw.A[j], AL))
    # ls[1] is the left half including site 0, so the l matrix needed at site 1.
    # ls[2] is the left half including site 1, so the l matrix needed at site 2.
    # etc...
    
    rs = [rR]
    for j in range(sw.N, 0, -1):
        rs.insert(0, tm.eps_r_noop(rs[0], sw.A[j], AR))
    # rs[0] is now the right half including site 1, so the r matrix needed
    # for computations involving site 0.

    rsBRs = [None]
    rsBRs += [
        [tm.eps_r_noop(rs[j3], sw.A[j3], BR) for BR in BRs]
        for j3 in range(1, len(rs))]
    rs = None
    
    # WARNING: This list will contain N^2/2 matrices!
    rsBRs_ext = []
    for j3, rBRs in enumerate(rsBRs):
        if rBRs is None:
            rsBRs_ext.append(None)
        else:
            rBRs_ext = [rBRs]
            # extend each rBRs to position 0 so that rBRs_ext[j3][j3] == rsBRs[j3]
            for j in reversed(range(j3)):
                if j == 0:
                    rBRs_ext.insert(0, None)
                else:
                    rBRs_ext.insert(0, [tm.eps_r_noop(rBR, sw.get_A(j), AC) for rBR in rBRs_ext[0]])
            rsBRs_ext.append(rBRs_ext)
    rsBRs = None

    num_ex = len(BLs)
    ols = []

    for j1 in range(1, N):
        ols.append([])
        l_mids = [tm.eps_l_noop(ls[j1], sw.A[j1], BL) for BL in BLs]
        ls[j1] = None
        for j2 in range(j1 + 1, N):
            ols[-1].append(np.zeros((N - j2, num_ex, num_ex, num_ex), dtype=np.complex128))
            l_mids_BC = [[tm.eps_l_noop(l, sw.A[j2], BC) for BC in BCs] for l in l_mids]
            l_mids = [tm.eps_l_noop(l, sw.A[j2], AC) for l in l_mids]
            for j3 in range(j2 + 1, N + 1):
                rBRs = rsBRs_ext[j3][j2 + 1]
                res = np.array([[[ma.adot(l, rBR) for rBR in rBRs] for l in l_BCs] for l_BCs in l_mids_BC])
                ols[-1][-1][j3 - j2 - 1, :,:,:] = res
                
                ## TODO: Avoid this part!
                #l_mids_BC = [[tm.eps_l_noop(l, sw.A[j3], AC) for l in l_BCs] for l_BCs in l_mids_BC]
            ols[-1][-1] = np.array(ols[-1][-1])

    return ols


def norms_3p_components_(sw, stateL, BLs, stateC, BCs, BRs, stateR, verbose=True):
    if not len(BLs) == len(BRs):
        raise ValueError("BLs and BRs have different lengths!")
    if not len(BLs) == len(BCs):
        raise ValueError("BLs and BCs have different lengths!")

    diffL = la.norm(stateL.A[0] - sw.uni_l.A[0])
    diffR = la.norm(stateR.A[0] - sw.uni_r.A[0])
    if diffL > 1e-10 or diffR > 1e-10:
        print("Warning! Bulk tensor differences: {}, {}".format(diffL, diffR))
    
    lL = stateL.l[0]
    rR = stateR.r[0]
       
    if not (stateL.L == 1 and stateR.L == 1 and stateC.L == 1):
        raise ValueError("Bulk unit cell size must currently be 1.")

    AR = stateR.A[0]
    for i, BR in enumerate(BRs):
        check_gf = tm.eps_r_noop(stateR.r[0], BR, AR)
        if la.norm(check_gf) > 1e-10:
            print("Warning! BRs[{}] RGF failed:".format(i), la.norm(check_gf))
        elif verbose:
            print("BRs[{}], check RGF:".format(i), la.norm(check_gf))

    AL = stateL.A[0]
    for i, BL in enumerate(BLs):
        check_gf = tm.eps_l_noop(stateL.l[0], BL, AL)
        if la.norm(check_gf) > 1e-10:
            print("Warning! BLs[{}] LGF failed:".format(i), la.norm(check_gf))
        elif verbose:
            print("BLs[{}], check LGF:".format(i), la.norm(check_gf))
    
    AC = stateC.A[0]
    
    N = sw.N
    
    ls = [lL, lL]
    for j in range(1, sw.N + 1):
        ls.append(tm.eps_l_noop(ls[-1], sw.A[j], AL))
    # ls[1] is the left half including site 0, so the l matrix needed at site 1.
    # ls[2] is the left half including site 1, so the l matrix needed at site 2.
    # etc...
    
    rs = [rR]
    for j in range(sw.N, 0, -1):
        rs.insert(0, tm.eps_r_noop(rs[0], sw.A[j], AR))
    # rs[0] is now the right half including site 1, so the r matrix needed
    # for computations involving site 0.

    rsBRs = [None]
    rsBRs += [
        [tm.eps_r_noop(rs[j2], sw.A[j2], BR) for BR in BRs]
        for j2 in range(1, len(rs))]
    rs = None
    
    rsBRs_ext = []
    for j3, rBRs in enumerate(rsBRs):
        if rBRs is None:
            rsBRs_ext.append(None)
        else:
            rBRs_ext = [rBRs]
            # extend each rBRs to position 0 so that rBRs_ext[j3][j3] == rsBRs[j3]
            for j in reversed(range(j3)):
                if j == 0:
                    rBRs_ext.insert(0, None)
                else:
                    rBRs_ext.insert(0, [tm.eps_r_noop(rBR, sw.get_A(j), AC) for rBR in rBRs_ext[0]])
            rsBRs_ext.append(rBRs_ext)
    rsBRs = None

    num_ex = len(BLs)
    probs = np.zeros((N, N, num_ex, num_ex, num_ex), dtype=np.complex128)

    for j1 in range(1, N):
        l_mids = [tm.eps_l_noop(ls[j1], sw.A[j1], BL) for BL in BLs]
        ls[j1] = None
        for j2 in range(j1 + 1, N):
            l_mids_BC = [[tm.eps_l_noop(l, sw.A[j2], BC) for BC in BCs] for l in l_mids]
            l_mids = [tm.eps_l_noop(l, sw.A[j2], AC) for l in l_mids]
            for j3 in range(j2 + 1, N + 1):
                rBRs = rsBRs_ext[j3][j2 + 1]
                res = np.array([[[ma.adot(l, rBR) for rBR in rBRs] for l in l_BCs] for l_BCs in l_mids_BC])
                probs[j2-j1,j3-j2, :,:,:] += np.abs(res)**2

    return np.sqrt(probs)


def basis_overlaps_3p_(stateL, BLs, stateC, BCs, BRs, stateR, max_d, verbose=True):
    """Overlaps for 2k,1p states.
    States are only nonorthogonal when the two kinks coincide. This computes all such
    overlaps (i.e. overlaps of different BC positions in case kinks coincide). We
    only compute overlaps for BC* coincident or to the right of BC. The other terms
    are just complex conjugates of these.
    Output: ols[d1, d2, d3].
    d1 is distance from BL to BC (> 0).
    d2 is distance from BC to BC* (>= 0).
    d3 is distance from BC* to BR.
    """
    if not (stateL.L == 1 and stateR.L == 1 and stateC.L == 1):
        raise ValueError("Bulk unit cell size must currently be 1.")
    
    # Let's just do this for single excitation tensors for now.
    assert len(BLs) == 1
    assert len(BRs) == 1
    assert len(BCs) == 1

    for i, B in enumerate(BRs):
        check_gf = tm.eps_r_noop(stateR.r[0], B, stateR.A[0])
        if la.norm(check_gf) > 1e-10:
            print("Warning! BR{} RGF failed: {}".format(i, la.norm(check_gf)))
        elif verbose:
            print("BR{}, check RGF: {}".format(i, la.norm(check_gf)))

    for i, B in enumerate(BLs):
        check_gf = tm.eps_l_noop(stateL.l[0], B, stateL.A[0])
        if la.norm(check_gf) > 1e-10:
            print("Warning! BL{} LGF failed: {}".format(i, la.norm(check_gf)))
        elif verbose:
            print("BL{}, check LGF: {}".format(i, la.norm(check_gf)))

    BL = BLs[0]
    BR = BRs[0]
    BC = BCs[0]
    l = tm.eps_l_noop(stateL.l[0], BL, BL)  # Not mixing BLs in this computation
    rs = [tm.eps_r_noop(stateR.r[0], BR, BR)]
    for _ in range(max_d):
        rs.append(tm.eps_r_noop(rs[-1], stateC.A[0], stateC.A[0]))
    
    ols = np.full((max_d, max_d, max_d), np.NaN, dtype=np.complex128)
    for d1 in range(1, max_d):
        l_BC = tm.eps_l_noop(l, BC, stateC.A[0])
        for d2 in range(max_d):
            if d2 == 0:
                l_2BC = tm.eps_l_noop(l, BC, BC)
            else:
                l_2BC = tm.eps_l_noop(l_BC, stateC.A[0], BC)
                l_BC = tm.eps_l_noop(l_BC, stateC.A[0], stateC.A[0])
            for d3 in range(1, max_d):
                ols[d1, d2, d3] = ma.adot(l_2BC, rs[d3-1])
                #l_2BC = tm.eps_l_noop(l_2BC, stateC.A[0], stateC.A[0])
        l = tm.eps_l_noop(l, stateC.A[0], stateC.A[0])

    return ols


def overlap_2p_components(sw, state_middle, p=0.0,
    verbose=True, force_pseudo=False, brute=True, num_ex=1):

    sw = copy.deepcopy(sw)
    sw.uni_l.phase_align(state_middle)
    sw.uni_r.phase_align(state_middle)
    sL = sw.uni_l
    sR = sw.uni_r
    sw.update()

    # if sw is not a TDVP, update() misses the following:
    sL.calc_C()
    sL.calc_K()
    sR.calc_C()
    sR.calc_K()

    B1s, B2s = excitation_tensors(
        sL, state_middle, sR, p=p, num_ex=num_ex,
        force_pseudo=force_pseudo, brute=brute, verbose=verbose)

    nrmsqs = basis_normalization_2p(
        sL, B1s, state_middle, B2s, sR, sw.N, verbose=verbose)

    ols = overlap_2p_components_(sw, sL, B1s, state_middle, B2s, sR, verbose=verbose) 

    # correct for small normalization differences
    for j1 in range(1, sw.N):
        for j2 in range(j1 + 1, sw.N + 1):
            ols[j1, j2, :, :] /= nrmsqs[j2 - j1 - 1, :, :]**0.5

    return ols, nrmsqs


def overlap_3p_components(sw, state_middle, p=0.0,
    verbose=True, force_pseudo=False, brute=True, num_ex=1, num_ex_C=1):
    sw = copy.deepcopy(sw)
    sw.uni_l.phase_align(state_middle)
    sw.uni_r.phase_align(state_middle)
    sL = sw.uni_l
    sR = sw.uni_r
    sw.update()

    # if sw is not a TDVP, update() misses the following:
    sL.calc_C()
    sL.calc_K()
    sR.calc_C()
    sR.calc_K()

    BLs, BRs = excitation_tensors(
        sL, state_middle, sR, p=p, num_ex=num_ex,
        force_pseudo=force_pseudo, brute=brute, verbose=verbose)
    
    ev, BCs = triv_excitations(state_middle, brute=brute, nev=num_ex_C)
    BCs = BCs[:num_ex_C]
    
    print(len(BLs), len(BRs), len(BCs))

    #nrmsqs = basis_normalization_2p(
    #    sL, B1s, state_middle, B2s, sR, sw.N, verbose=verbose)

    ols = overlap_3p_components_(sw, sL, BLs, state_middle, BCs, BRs, sR, verbose=verbose) 

    ## correct for small normalization differences
    #for j1 in range(1, sw.N):
    #    for j2 in range(j1 + 1, sw.N + 1):
    #        ols[j1, j2, :, :] /= nrmsqs[j2 - j1 - 1, :, :]**0.5

    return ols


def norms_3p_components(sw, state_middle, p=0.0,
    verbose=True, force_pseudo=False, brute=True, num_ex=1, num_ex_C=1):
    sw = copy.deepcopy(sw)
    sw.uni_l.phase_align(state_middle)
    sw.uni_r.phase_align(state_middle)
    sL = sw.uni_l
    sR = sw.uni_r
    sw.update()

    # if sw is not a TDVP, update() misses the following:
    sL.calc_C()
    sL.calc_K()
    sR.calc_C()
    sR.calc_K()

    BLs, BRs = excitation_tensors(
        sL, state_middle, sR, p=p, num_ex=num_ex,
        force_pseudo=force_pseudo, brute=brute, verbose=verbose)
    
    ev, BCs = triv_excitations(state_middle, brute=brute, nev=num_ex_C)
    BCs = BCs[:num_ex_C]
    
    print(len(BLs), len(BRs), len(BCs))

    #nrmsqs = basis_normalization_2p(
    #    sL, B1s, state_middle, B2s, sR, sw.N, verbose=verbose)

    norms = norms_3p_components_(sw, sL, BLs, state_middle, BCs, BRs, sR, verbose=verbose) 

    ## correct for small normalization differences
    #for j1 in range(1, sw.N):
    #    for j2 in range(j1 + 1, sw.N + 1):
    #        ols[j1, j2, :, :] /= nrmsqs[j2 - j1 - 1, :, :]**0.5

    return norms


def basis_overlaps_3p(sw, state_middle, d_max, p=0.0,
    verbose=True, force_pseudo=False, brute=True, num_ex=1, num_ex_C=1):
    sw = copy.deepcopy(sw)
    sw.uni_l.phase_align(state_middle)
    sw.uni_r.phase_align(state_middle)
    sL = sw.uni_l
    sR = sw.uni_r
    sw.update()

    # if sw is not a TDVP, update() misses the following:
    sL.calc_C()
    sL.calc_K()
    sR.calc_C()
    sR.calc_K()

    BLs, BRs = excitation_tensors(
        sL, state_middle, sR, p=p, num_ex=num_ex,
        force_pseudo=force_pseudo, brute=brute, verbose=verbose)
    
    ev, BCs = triv_excitations(state_middle, brute=brute, nev=num_ex_C)
    BCs = BCs[:num_ex_C]
    
    print(len(BLs), len(BRs), len(BCs))

    bolps = basis_overlaps_3p_(sL, BLs, state_middle, BCs, BRs, sR, d_max, verbose=verbose) 

    return bolps


def orthonormalize(Bs, sL, sR):
    l = sL.l[0]
    r = sR.r[0]

    def inner(B1, B2):
        return ma.adot(l, tm.eps_r_noop(r, B2, B1))

    Bs_orth = [Bs[0] / np.sqrt(inner(Bs[0], Bs[0]))]
    for i in range(1, len(Bs)):
        newB = Bs[i] - sum([B * inner(B, Bs[i]) for B in Bs_orth[:i]])
        newB /= np.sqrt(inner(newB, newB))
        Bs_orth.append(newB)

    #chk = [abs(inner(B1,B2)) for B1 in Bs_orth for B2 in Bs_orth]
    #print(chk)

    return Bs_orth


def _prep_2p_Bs(sL, state_middle, sR, ps=[0.0], num_ex=1,
    cross_terms=True, ortho=True,
    verbose=True, force_pseudo=False, brute=True):

    ex_labels = []
    B1s = []
    B2s = []
    for p in ps:
        _B1s, _B2s = excitation_tensors(
            sL, state_middle, sR, p=p, num_ex=num_ex,
            force_pseudo=force_pseudo, brute=brute, verbose=verbose)
        B1s.extend(_B1s[:num_ex])
        B2s.extend(_B2s[:num_ex])
        ex_labels = ex_labels + [(p, i) for i in range(num_ex)]

    if len(ps) > 1 and ortho:
        B1s = orthonormalize(B1s, sL, state_middle)
        B2s = orthonormalize(B2s, state_middle, sR)

    if cross_terms:
        # NOTE: Need cross-terms to get everything.
        # Functions below match entries in B1s and B2s by index (a la `zip`).
        # To get cross-terms, we need to duplicate entries as follows.
        B1s = sum([[B1s[i] for _ in range(len(B1s))] for i in range(len(B1s))], [])
        B2s = sum([list(B2s) for _ in range(len(B2s))], [])
        ex_labels = [(l, m) for l in ex_labels for m in ex_labels]

    return B1s, B2s, ex_labels


def _prep_2p(sw, state_middle, ps=[0.0], num_ex=1,
    cross_terms=True, ortho=True,
    verbose=True, force_pseudo=False, brute=True):
    sw = copy.deepcopy(sw)
    sw.uni_l.phase_align(state_middle)
    sw.uni_r.phase_align(state_middle)
    sL = sw.uni_l
    sR = sw.uni_r
    sw.update()

    # if sw is not a TDVP, update() misses the following:
    sL.calc_C()
    sL.calc_K()
    sR.calc_C()
    sR.calc_K()

    B1s, B2s, ex_labels = _prep_2p_Bs(
        sL, state_middle, sR,
        ps=ps, num_ex=num_ex,
        cross_terms=cross_terms, ortho=ortho,
        verbose=verbose, force_pseudo=force_pseudo, brute=brute)
    
    return sw, sL, sR, B1s, B2s, ex_labels


def overlap_2p_components_multip(sw, state_middle, ps=[0.0], num_ex=1, ortho=True,
    verbose=True, force_pseudo=False, brute=True, return_prep_data=False):
    """Projects onto a (separated) 2-particle position basis.

    Only the state to be projected `sw` and the uniform MPS representing the
    state between the two particles `state_middle`, are needed. The excitation
    tensors B1 and B2 are computed from bulk states of `sw` together with
    `state_middle`.

    The output is a matrix, with one axis 0 the position of the left particle,
    and axis 1 the position of the right particle. Disallowed combinations of
    position (with the right particle to the left of the left particle) are
    filled in with NaN.
    """
    sw, sL, sR, B1s, B2s, ex_labels = _prep_2p(
        sw, state_middle, ps=ps, num_ex=num_ex,
        cross_terms=False, ortho=ortho,
        verbose=verbose, force_pseudo=force_pseudo, brute=brute)

    # FIXME: Should really compute the gram matrix here, since there is no
    #        guarantee that the two-particle states are orthogonal, at least
    #        at small separations. At large separations, the transfer-matrix
    #        gap should kick in, and the overlaps between these states should
    #        correspond to the 1-B overlaps.
    nrmsqs = basis_normalization_2p(
                sL, B1s, state_middle, B2s, sR, sw.N, verbose=verbose)

    ols = overlap_2p_components_(sw, sL, B1s, state_middle, B2s, sR, verbose=verbose) 

    # correct for small normalization differences
    for j1 in range(1, sw.N):
        for j2 in range(j1 + 1, sw.N + 1):
            ols[j1, j2, :, :] /= nrmsqs[j2 - j1 - 1, :, :]**0.5

    if return_prep_data:
        return ols, nrmsqs, ex_labels, sL, sR, B1s, B2s
    return ols, nrmsqs, ex_labels


def basis_overlaps_2p_multip(sw, state_middle, ps=[0.0], num_ex=1,
    ortho=True,
    verbose=True, force_pseudo=False, brute=True):
    """Projects onto a (separated) 2-particle position basis.

    Only the state to be projected `sw` and the uniform MPS representing the
    state between the two particles `state_middle`, are needed. The excitation
    tensors B1 and B2 are computed from bulk states of `sw` together with
    `state_middle`.

    The output is a matrix, with one axis 0 the position of the left particle,
    and axis 1 the position of the right particle. Disallowed combinations of
    position (with the right particle to the left of the left particle) are
    filled in with NaN.
    """
    sw, sL, sR, BLs, BRs, ex_labels = _prep_2p(
        sw, state_middle, ps=ps, num_ex=num_ex,
        cross_terms=False, ortho=ortho,
        verbose=verbose, force_pseudo=force_pseudo, brute=brute)

    ols = basis_overlaps_2p_(
                sL, BLs, state_middle, BRs, BLs, state_middle, BRs, sR, sw.N, verbose=verbose)

    return ols, ex_labels


def basis_overlaps_2p_multip_cross(sw, ex_specs,
    ortho=True,
    verbose=True, brute=True):

    assert len(ex_specs) == 2
    
    state_middle0, ps0, num_ex0, force_pseudo0 = ex_specs[0]

    sw, sL, sR, BLs, BRs, ex_labels = _prep_2p(
        sw, state_middle0, ps=ps0, num_ex=num_ex0,
        cross_terms=False, ortho=ortho,
        verbose=verbose, force_pseudo=force_pseudo0, brute=brute)

    ex_Bdata = [(BLs, BRs, ex_labels)]

    for ex_spec in ex_specs[1:]:
        state_middle, ps, num_ex, force_pseudo = ex_spec
        state_middle = copy.deepcopy(state_middle)
        state_middle.phase_align(state_middle0)

        BLs, BRs, _ex_labels = _prep_2p_Bs(
            sL, state_middle, sR,
            ps=ps, num_ex=num_ex,
            cross_terms=False, ortho=ortho,
            verbose=verbose, force_pseudo=force_pseudo, brute=brute)
        
        ex_Bdata.append((BLs, BRs, ex_labels))

    ols = basis_overlaps_2p_(
      sL,
      ex_Bdata[0][0], ex_specs[0][0], ex_Bdata[0][1],
      ex_Bdata[1][0], ex_specs[1][0], ex_Bdata[1][1],
      sR, sw.N, verbose=verbose)

    return ols, ex_Bdata[0][2], ex_Bdata[1][2]


def basis_overlaps_2p(sw, state_middle1, state_middle2, p1=0.0, p2=0.0, verbose=True,
                      force_pseudo1=False, force_pseudo2=False, brute=True):
    sw = copy.deepcopy(sw)
    sw.uni_l.phase_align(state_middle1)
    sw.uni_r.phase_align(state_middle1)
    sL = sw.uni_l
    sR = sw.uni_r
    sw.update()

    state_middle2 = copy.deepcopy(state_middle2)
    state_middle2.phase_align(state_middle1)

    # if sw is not a TDVP, update() misses the following:
    sL.calc_C()
    sL.calc_K()
    sR.calc_C()
    sR.calc_K()

    BL1, BR1 = excitation_tensors(
        sL, state_middle1, sR, p=p1, force_pseudo=force_pseudo1, brute=brute, verbose=verbose)
    BL2, BR2 = excitation_tensors(
        sL, state_middle2, sR, p=p2, force_pseudo=force_pseudo2, brute=brute, verbose=verbose)

    nrmsqs1 = basis_normalization_2p(sL, BL1[:1], state_middle1, BR1[:1], sR, sw.N, verbose=verbose)
    nrmsqs2 = basis_normalization_2p(sL, BL2[:1], state_middle2, BR2[:1], sR, sw.N, verbose=verbose)

    ols = basis_overlaps_2p_(
        sL, BL1[:1], state_middle1, BR1[:1], BL2[:1], state_middle2, BR2[:1], sR, sw.N, verbose=verbose)

    # correct for small normalization differences
    for j in range(len(ols)):
        ols[j] /= (nrmsqs1[j] * nrmsqs2[j])**0.5

    return ols, nrmsqs1, nrmsqs2


def overlap_1p_components_triv(state, num_ex, brute=True, verbose=True):
    """Projects onto a 1-particle basis of (topologically trivial) states.

    These are defined to be orthogonal to the 2-particle basis used above.

    The 1-particle basis states are obtained variationally as excitations
    using the left and right bulk states of the input state as the "vacuum".
    If the left and right bulk states are (physically) equivalent, these are
    topologically trivial excitations.

    Args:
      state: The input state to project onto the 1-particle basis.
      num_ex: The number of topologically nontrivial excitations to include
        in the basis.
    Returns:
      A list, length `num_ex`, of overlap vectors, one for each topologically
      trivial excitation.
    """
    state.uni_l.calc_K()  # ensure the energy has been calculated!
    ev, Bs_triv = top_excitations(state.uni_l, state.uni_r, force_pseudo=True, brute=brute, nev=num_ex)
    if verbose: print(ev)
    ols = overlap_1p_components_(state, Bs_triv[:num_ex], verbose=verbose)
    return ols


def overlap_1p_components_(state, Bs, verbose=True):
    """Inner product of the state with an MPS tangent vector components.

    B tensors must match the left and right uniform bulk tensors of state!
    
    Output axes: (Excitation number, position)
    """
    if not (state.uni_l.L == 1 and state.uni_r.L == 1):
        raise ValueError("Bulk unit cell size must currently be 1.")
    rR = state.uni_r.r[0]
    AR = state.uni_r.A[0]
    for i, B in enumerate(Bs):
        check_gf = tm.eps_r_noop(rR, AR, B)
        if la.norm(check_gf) > 1e-10:
            print("Warning! Bs[{}] LGF failed:".format(i), la.norm(check_gf))
        elif verbose:
            print("Bs[{}], check LGF:".format(i), la.norm(check_gf))

    lL = state.uni_l.l[0]
    AL = state.uni_l.A[0]

    rs = [rR]
    for j in range(state.N, 0, -1):
        rs.insert(0, tm.eps_r_noop(rs[0], state.A[j], AR))
    # rs[0] is now the right half including site 1, so the r matrix needed
    # for computations involving site 0.

    l = lL
    ols = [[] for _ in Bs]
    for j in range(1, state.N + 1):
        for i, B in enumerate(Bs):
            res = ma.adot(l, tm.eps_r_noop(rs[j], state.A[j], B))
            ols[i].append(res)
        rs[j] = None  # free
        l = tm.eps_l_noop(l, state.A[j], AL)

    return np.array(ols)


def load_state(filename, ham=None, do_update=True):
    """Loads a sandwich state and adds a Hamiltonian for the uniform bulk parts.
    This is sufficient for the resulting sandwich MPS object to be used to
    compute the overlaps above.
    """
    if filename.endswith(".pickle"):
        with open(filename, "rb") as f:
            s = pickle.load(f)
    else:
        s = mps_s.EvoMPS_MPS_Sandwich.from_file(filename)

    if ham is not None:
        s.update(restore_CF=False)
        #s = tdvp_s.EvoMPS_TDVP_Sandwich.from_mps(s, ham, [ham] * (s.N + 1), ham)
        s.uni_l = tdvp_u.EvoMPS_TDVP_Uniform.from_mps(s.uni_l, ham)
        s.uni_r = tdvp_u.EvoMPS_TDVP_Uniform.from_mps(s.uni_r, ham)
    if do_update:
        s.update()
        if ham is not None:
            s.uni_l.calc_C()
            s.uni_l.calc_K()
            s.uni_r.calc_C()
            s.uni_r.calc_K()
    return s


def state_fn(dt, t):
    ind = int(t / dt) - 1
    return "state_evo_step{}.npy".format(ind)


def wf_sep(psi, d_min=30):
    psi_sep = psi.copy()
    for j in range(psi.shape[0]):
        for k in range(j, min(psi.shape[1], j+d_min)):
            psi_sep[j,k] = np.NaN
    return psi_sep


def dt_str(dt):
    if dt >= 1.0:
        raise ValueError("Don't know how to represent {}".format(dt))
    return ("%.10f" % dt)[2:].rstrip('0')


def getdir(basedir, D, dt, intg):
    return "{}/maxD{}_dt{}{}/".format(basedir, D, dt_str(dt), intg)


def get_ham(J, hx, hz=0.0, J2=0.0, J3=0.0, J4=0.0, J5=0.0, hzx=0.0):
  Sx = np.array(
    [[0, 1],
     [1, 0]])
  Sy = 1.j * np.array(
    [[0, -1],
     [1, 0]])
  Sz = np.array(
    [[1, 0],
     [0, -1]])
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


def load_ham(basedir):
    with open(basedir + "/state_params.json", "r") as f:
        state_params = json.load(f)
    ham = get_ham(
        state_params["J"],
        state_params["hx"],
        state_params["hz"],
        state_params["J2"],
        state_params["J3"],
        state_params["J4"],
        state_params["J5"],
        state_params["hzx"])
    return ham


def scan_states(path, interval, t_max):
    with open(path + "/evo_params.json", "r") as f:
        evo_params = json.load(f)
    
    dt = evo_params["dt"]
    num = int(t_max // interval)
    ts = [interval*i for i in range(num + 1)]

    if not ts:
        return

    yield (0, path + "initial_state.pickle")

    for t in ts[1:]:
        fn = path + state_fn(dt, t)
        yield (t, fn)


def scan_states_1p(path, interval, t_max, num_ex=10):
    ham = load_ham(path)

    norms = []
    ts = []
    for (t, fn) in scan_states(path, interval, t_max):
        print("Starting:", t)
        state = load_state(fn, ham=ham)
        # each step takes O(N) time
        psi = overlap_1p_components_triv(state, num_ex, verbose=False)
        nrm = la.norm(np.nan_to_num(psi), axis=1)
        print("Done:", nrm)
        norms.append(nrm)
        ts.append(t)
        np.save(path + "norms_1p_nex{}.npy".format(num_ex), norms)
        np.save(path + "ts_1p_nex{}.npy".format(num_ex), ts)
        np.save(path + "psi_1p_nex{}_t{}.npy".format(num_ex, t), psi)


def scan_states_2k(path, interval, t_max, ps=[0.0, -0.4, 0.4], num_ex=1):
    ham = load_ham(path)

    falsevac = tdvp_u.EvoMPS_TDVP_Uniform.from_file(
        path + "/../vac_in_uniform.npy", ham)

    norms = []
    ts = []
    for (t, fn) in scan_states(path, interval, t_max):
        print("Starting:", t)
        state = load_state(fn, ham=ham)
        # each step takes O(N^2) time
        psi, _, exlbls = overlap_2p_components_multip(
            state, falsevac, ps=ps, num_ex=num_ex, ortho=True, brute=True,
            verbose=False)
        nrm = la.norm(np.nan_to_num(psi), axis=(0,1))
        print("Done:", nrm)
        norms.append(nrm)
        ts.append(t)
        np.save(path + "norms_2k_nex{}_np{}.npy".format(num_ex, len(ps)), norms)
        np.save(path + "ts_2k_nex{}_np{}.npy".format(num_ex, len(ps)), ts)
        np.save(path + "psi_2k_nex{}_np{}_t{}.npy".format(num_ex, len(ps), t), psi)
        np.save(path + "exlbls_2k_nex{}_np{}.npy".format(num_ex, len(ps)), exlbls)


def scan_states_2p(path, interval, t_max, ps=[0.0, -0.6], num_ex=3):
    ham = load_ham(path)

    vac = tdvp_u.EvoMPS_TDVP_Uniform.from_file(
        path + "/../vac_out_uniform.npy", ham)

    norms = []
    ts = []
    for (t, fn) in scan_states(path, interval, t_max):
        print("Starting:", t)
        state = load_state(fn, ham=ham)
        # each step takes O(N^2) time
        psi, _, exlbls = overlap_2p_components_multip(
            state, vac, ps=ps, num_ex=num_ex, ortho=True, cross_terms=True,
            brute=True, force_pseudo=True, verbose=False)
        nrm = la.norm(np.nan_to_num(psi), axis=(0,1))
        print("Done:", nrm)
        norms.append(nrm)
        ts.append(t)
        np.save(path + "norms_2p_nex{}_np{}.npy".format(num_ex, len(ps)), norms)
        np.save(path + "ts_2p_nex{}_np{}.npy".format(num_ex, len(ps)), ts)
        np.save(path + "psi_2p_nex{}_np{}_t{}.npy".format(num_ex, len(ps), t), psi)
        np.save(path + "exlbls_2p_nex{}_np{}.npy".format(num_ex, len(ps)), exlbls)
