import shutil, os, sys, json

dirname = str(sys.argv[1])
jobprefix = str(sys.argv[2])
hx = 0.4
hz = 0.01
Dvac = 5
N = 348
pkt_numsites = 72
pkt_sigma = 7.0
pkt_midpt = 36.0
pad_mid = 0
pad_out = 100

masterpath = os.path.join(os.getcwd(), dirname)

if os.path.exists(masterpath):
    raise ValueError("master dir. '{}' exists!".format(masterpath))

os.makedirs(masterpath)

state_params = {
    "J": 1.0,
    "hx": hx,
    "hz": hz,
    "J2": 0.0,
    "J3": 0.0,
    "J4": 0.0,
    "J5": 0.0,
    "hzx": 0.0,
    "D": Dvac,
    "N": N,
    "ortho_2p": False,
    "symmB": False,
    "pkt_numsites": pkt_numsites,
    "pkt_sigma": pkt_sigma,
    "pkt_midpt": pkt_midpt,
    "pad_mid": pad_mid,
    "pad_out": pad_out,
    "truevac_outer": True,
    "truevac_inner": False,
    "momentum": 0,
    "kink_lvl": 0,
}
with open(os.path.join(masterpath, "./state_params.json"), "w") as f:
    json.dump(state_params, f)

job_params = {
    "jobprefix": jobprefix,
}
with open(os.path.join(masterpath, "./job_params.json"), "w") as f:
    json.dump(job_params, f)

os.symlink("/home/amilsted/evoMPS/evoMPS", os.path.join(masterpath, "./evoMPS"))
os.symlink("/home/amilsted/TensorNetwork/tensornetwork", os.path.join(masterpath, "./tensornetwork"))

shutil.copyfile("sources/master.py", os.path.join(masterpath, "./master.py"))
shutil.copyfile("sources/bsymm.py", os.path.join(masterpath, "./bsymm.py"))
shutil.copyfile("sources/make_jobdir.py", os.path.join(masterpath, "./make_jobdir.py"))
shutil.copyfile("sources/collect_data.py", os.path.join(masterpath, "./collect_data.py"))
