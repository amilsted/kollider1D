import shutil, os, sys, json
import numpy as np

system = "slurm"

from_dir = sys.argv[1]
to_dir = sys.argv[2]
t_cut = int(sys.argv[3])

submit_job = len(sys.argv) == 5 and str(sys.argv[4]) == "--submit"

with open(from_dir + "/evo_params.json", "r") as f:
    fromevo_params = json.load(f)
    
with open(to_dir + "/evo_params.json", "r") as f:
    toevo_params = json.load(f)
    
from_dt = fromevo_params["dt"]
to_dt = toevo_params["dt"]

if from_dt != to_dt:
    raise ValueError("dt's do not match!")
    
stepnum_cut = int(t_cut / from_dt) - 1
print("Copying up to step:", stepnum_cut)

expvals_evo = list(np.load(from_dir + "/expvals_evo.npy", allow_pickle=True))[:stepnum_cut+1]
Ds_evo = list(np.load(from_dir + "/Ds_evo.npy", allow_pickle=True))[:stepnum_cut+1]
eta_sqs = list(np.load(from_dir + "/eta_sqs_evo.npy", allow_pickle=True))[:stepnum_cut+1]
schmidts = list(np.load(from_dir + "/schmidts_evo.npy", allow_pickle=True))[:stepnum_cut+1]
proj_errs = list(np.load(from_dir + "/proj_errs_evo.npy", allow_pickle=True))[:stepnum_cut+1]

np.save(to_dir + "/expvals_evo.npy", np.vstack(expvals_evo))
np.save(to_dir + "/Ds_evo.npy", np.vstack(Ds_evo))
np.save(to_dir + "/eta_sqs_evo.npy", eta_sqs)
np.save(to_dir + "/schmidts_evo.npy", schmidts)
np.save(to_dir + "/proj_errs_evo.npy", proj_errs)

statefn = "/state_evo_step{}.npy".format(stepnum_cut)
shutil.copyfile(from_dir + statefn, to_dir + statefn)

shutil.copyfile(from_dir + "/evo_params.json", to_dir + "/evo_params_upto_t{}.json".format(t_cut))

if submit_job:
    jobpath = os.path.join(os.getcwd(), to_dir)

    import subprocess
    if system == "slurm":
        subprocess.run(["sbatch", os.path.join(jobpath, "./jobscript.sh")])
    else:
        subprocess.run(["qsub", os.path.join(jobpath, "./jobscript.sh")])
