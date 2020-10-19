import numpy as np
import scipy.linalg as la
import tensornetwork as tn


def gf_cost_func_v1(B, AL, AR):
    Bn = tn.Node(B, axis_names=["p", "L", "R"])
    ALn = tn.Node(AL, axis_names=["p", "L", "R"])
    ARn = tn.Node(AR, axis_names=["p", "L", "R"])
    
    ALn_c = tn.conj(ALn)
    ALn["L"] ^ ALn_c["L"]
    ALn["R"] ^ ALn_c["R"]
    PLn = tn.contract_between(ALn, ALn_c, output_edge_order=[ALn_c["p"], ALn["p"]], axis_names=["p_in", "p_out"])
    
    ARn_c = tn.conj(ARn)
    ARn["L"] ^ ARn_c["L"]
    ARn["R"] ^ ARn_c["R"]
    PRn = tn.contract_between(ARn, ARn_c, output_edge_order=[ARn_c["p"], ARn["p"]], axis_names=["p_in", "p_out"])
    
    PLRn = tn.Node(PLn.get_tensor() + PRn.get_tensor(), axis_names=["p_in", "p_out"])
    
    Bn_c = tn.conj(Bn)
    Bn["L"] ^ Bn_c["L"]
    Bn["R"] ^ Bn_c["R"]
    Bn["p"] ^ PLRn["p_in"]
    Bn_c["p"] ^ PLRn["p_out"]
    res = Bn @ PLRn @ Bn_c
    return res.get_tensor()


def regauge_B_symm_linear_problem_v1(B, AL, AR, p=0):
    Bn = tn.Node(B, axis_names=["p", "L", "R"])
    ALn = tn.Node(AL, axis_names=["p", "L", "R"])
    ARn = tn.Node(AR * np.exp(-1.j*p), axis_names=["p", "L", "R"])
    
    D = B.shape[2]
    
    ALn_c = tn.conj(ALn)
    ALn["L"] ^ ALn_c["L"]
    ALn["R"] ^ ALn_c["R"]
    PLn = tn.contract_between(ALn, ALn_c, output_edge_order=[ALn_c["p"], ALn["p"]], axis_names=["p_in", "p_out"])
    
    ARn_c = tn.conj(ARn)
    ARn["L"] ^ ARn_c["L"]
    ARn["R"] ^ ARn_c["R"]
    PRn = tn.contract_between(ARn, ARn_c, output_edge_order=[ARn_c["p"], ARn["p"]], axis_names=["p_in", "p_out"])
    
    PLRn = tn.Node(PLn.get_tensor() + PRn.get_tensor(), axis_names=["p_in", "p_out"])
    
    E = tn.CopyNode(2, D, dtype=B.dtype)
    
    PLRn["p_out"] ^ ALn_c["p"]
    MLn = tn.contract_between(
        PLRn, ALn_c,
        output_edge_order=[PLRn["p_in"], ALn_c["L"], ALn_c["R"]],
        axis_names=["p_in", "L_in", "L_out"])
    MLn = tn.contract_between(
        MLn, E,
        output_edge_order=[MLn["L_out"], E[0], MLn["p_in"], MLn["L_in"], E[1]],
        axis_names=["L_out", "R_out", "p_in", "L_in", "R_in"],
        allow_outer_product=True)
    
    PLRn["p_out"] ^ ARn_c["p"]
    MRn = tn.contract_between(
        PLRn, ARn_c,
        output_edge_order=[PLRn["p_in"], ARn_c["L"], ARn_c["R"]],
        axis_names=["p_in", "R_out", "R_in"])
    MRn = tn.contract_between(
        MRn, E,
        output_edge_order=[E[0], MRn["R_out"], MRn["p_in"], E[1], MRn["R_in"]],
        axis_names=["L_out", "R_out", "p_in", "L_in", "R_in"],
        allow_outer_product=True)
    
    MLRn = tn.Node(
        MLn.get_tensor() - MRn.get_tensor(),
        axis_names=["L_out", "R_out", "p_in", "L_in", "R_in"])
    
    MLRn["p_in"] ^ Bn["p"]
    MLRn["L_in"] ^ Bn["L"]
    MLRn["R_in"] ^ Bn["R"]
    target_n = tn.contract_between(MLRn, Bn, output_edge_order=[MLRn["L_out"], MLRn["R_out"]])
    target_vec = target_n.get_tensor().ravel()
    
    ARn["p"] ^ MLRn["p_in"]
    ARn["R"] ^ MLRn["R_in"]
    
    bigM_Rn = tn.contract_between(
        ARn, MLRn,
        output_edge_order=[MLRn["L_out"], MLRn["R_out"], MLRn["L_in"], ARn["L"]])
    
    ALn["p"] ^ MLRn["p_in"]
    ALn["L"] ^ MLRn["L_in"]
    
    bigM_Ln = tn.contract_between(
        ALn, MLRn,
        output_edge_order=[MLRn["L_out"], MLRn["R_out"], ALn["R"], MLRn["R_in"]])
    
    bigMn = tn.Node(
        bigM_Rn.get_tensor() - bigM_Ln.get_tensor(),
        axis_names=["L_out", "R_out", "L_in", "R_in"])
    mat = bigMn.get_tensor().reshape((D**2, D**2))
    
    return mat, target_vec


def inner(B1, B2, stateL, stateR):
  B1n = tn.Node(B1, axis_names=["p", "L", "R"])
  B2nc = tn.conj(tn.Node(B2, axis_names=["p", "L", "R"]))
  ln = tn.Node(np.asarray(stateL.l[0]), axis_names=["B", "T"])
  rn = tn.Node(np.asarray(stateR.r[0]), axis_names=["T", "B"])
  B1n["p"] ^ B2nc["p"]
  B1n["L"] ^ ln["T"]
  B2nc["L"] ^ ln["B"]
  B1n["R"] ^ rn["T"]
  B2nc["R"] ^ rn["B"]
  return (ln @ B1n @ rn @ B2nc).get_tensor()


def symmetrize_B(B, stateL, stateR, p=0, force_orth_vac=False):
  mat, vec = regauge_B_symm_linear_problem_v1(B, stateL.A[0], stateR.A[0], p=p)
  Xvec, resid, rank, _ = la.lstsq(mat, vec)
  #print("residual:", resid, "rank:", rank)
  X = Xvec.reshape(stateL.D, stateL.D)
  Bnew = np.array(
      [B[t] + stateL.A[0][t].dot(X) - np.exp(-1.j*p) * X.dot(stateR.A[0][t])
       for t in range(B.shape[0])])
  nrm2 = np.sqrt(inner(Bnew, Bnew, stateL, stateR))
  c1 = gf_cost_func_v1(B, stateL.A[0], stateR.A[0])
  c2 = gf_cost_func_v1(Bnew, stateL.A[0], stateR.A[0])
  if force_orth_vac:
    # In case stateL and stateR are the same, and p is nonzero, we have the
    # additional freedom to add a factor of the vacuum tensor. Use it to make
    # individual terms orthogonal to the vacuum, as they are for LGF or RGF.
    if p == 0 or la.norm(stateL.A[0] - stateR.A[0]) > 1e-10:
        print("WARNING! force_orth_vac does not make sense here!")
    olA = inner(Bnew, stateL.A[0], stateL, stateR)
    Bnew -= olA * stateL.A[0]
    print("vac overlap:", olA, inner(Bnew, stateL.A[0], stateL, stateR))
  c3 = gf_cost_func_v1(Bnew, stateL.A[0], stateR.A[0])
  nrm3 = np.sqrt(inner(Bnew, Bnew, stateL, stateR))
  print("'norms':", inner(B, B, stateL, stateR).real, nrm2.real, nrm3.real)
  print("costs:", c1.real, c2.real, c3.real)
  #Bnew /= nrm
  return Bnew


def test(B, stateL, stateR, p):
  mat, vec = regauge_B_symm_linear_problem_v1(B, stateL.A[0], stateR.A[0], p=p)
  Xvec, resid, rank, _ = la.lstsq(mat, vec)
  X = Xvec.reshape(stateL.D, stateL.D)

  Bnew = np.array(
      [B[t] + stateL.A[0][t].dot(X) - np.exp(-1.j*p) * X.dot(stateR.A[0][t])
       for t in range(B.shape[0])])
  c1 = gf_cost_func_v1(Bnew, stateL.A[0], stateR.A[0])

  Bnew = np.array(
      [B[t] + stateL.A[0][t].dot(X) - np.exp(1.j*p) * X.dot(stateR.A[0][t])
       for t in range(B.shape[0])])
  c2 = gf_cost_func_v1(Bnew, stateL.A[0], stateR.A[0])
  return c1.real, c2.real
