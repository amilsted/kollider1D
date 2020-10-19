import shutil, os, sys, json, pickle
import numpy as np
import master as ma

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
    
stepnum_cut = int(t_cut / from_dt) - 1
print("Copying step", stepnum_cut, "to initial state.")

statefn = "/state_evo_step{}.npy".format(stepnum_cut)

with open("initial_state.pickle", "rb") as f:
    swp = pickle.load(f)
swp = ma.load_state(from_dir + statefn, ham_uni=swp.uni_l.ham)

with open(to_dir + "/initial_state.pickle", "wb") as f:
    pickle.dump(swp, f)
    
shutil.copyfile(from_dir + "/evo_params.json", to_dir + "/evo_params_upto_t{}_asinitial.json".format(t_cut))

os.symlink("../" + from_dir, to_dir + "/initial_from")

if submit_job:
    jobpath = os.path.join(os.getcwd(), to_dir)

    import subprocess
    if system == "slurm":
        subprocess.run(["sbatch", os.path.join(jobpath, "./jobscript.sh")])
    else:
        subprocess.run(["qsub", os.path.join(jobpath, "./jobscript.sh")])
