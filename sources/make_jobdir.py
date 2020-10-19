import shutil, os, sys, json

system = "slurm"

Dmax = int(sys.argv[1])
dt = float(sys.argv[2])
integrator = str(sys.argv[3])
t_end = float(sys.argv[4])
n_cores = int(sys.argv[5])
mem = int(sys.argv[6])

submit_job = len(sys.argv) == 8 and str(sys.argv[7]) == "--submit"

with open("job_params.json", "r") as f:
    job_params = json.load(f)
jobprefix = job_params["jobprefix"]

print(
    "Dmax = {}, dt = {}, int. = {}, t_end = {}".format(
        Dmax, dt, integrator, t_end))

def dt_str(dt):
    if dt >= 1.0:
        raise ValueError("Don't know how to represent {}".format(dt))
    return ("%.10f" % dt)[2:].rstrip('0')

dtstr = dt_str(dt)
suff = "" if integrator == "split" else integrator
dirname = "maxD{}_dt{}{}/".format(Dmax, dtstr, suff)

jobpath = os.path.join(os.getcwd(), dirname)

if os.path.exists(jobpath):
    raise ValueError("job dir. '{}' exists!".format(jobpath))

os.makedirs(jobpath)

jobname = "{}_D{}_dt{}{}".format(jobprefix, Dmax, dtstr, suff)

jobscript_pbs = """#!/bin/sh
#PBS -N {}
#PBS -l nodes=1:ppn={}
#PBS -l walltime=200:00:00
#PBS -l mem={}gb

export OMP_NUM_THREADS={}

cd {}
python sim.py evolve
""".format(jobname, n_cores, mem, n_cores, jobpath)

jobscript_slurm = """#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH --ntasks={}
#SBATCH --nodes=1
#SBATCH --mem={}G
#SBATCH -J "{}"
#SBATCH -o "{}_%j.out"
#SBATCH -e "{}_%j.err"

export OMP_NUM_THREADS={}

cd {}
python sim.py evolve
""".format(n_cores, mem, jobname, jobname, jobname, n_cores, jobpath)

with open(os.path.join(jobpath, "./jobscript.sh"), "w") as f:
    f.write(jobscript_slurm if system == "slurm" else jobscript_pbs)

evo_params = {
    "integrator": integrator,
    "Dmax": Dmax,
    "dt": dt,
    "t_end": t_end
}
with open(os.path.join(jobpath, "./evo_params.json"), "w") as f:
    json.dump(evo_params, f)

os.symlink("/home/amilsted/evoMPS/evoMPS", os.path.join(jobpath, "./evoMPS"))

shutil.copyfile("master.py", os.path.join(jobpath, "./sim.py"))

shutil.copyfile("initial_state.pickle", os.path.join(jobpath, "./initial_state.pickle"))

shutil.copyfile("state_params.json", os.path.join(jobpath, "./state_params.json"))

if submit_job:
    import subprocess
    if system == "slurm":
        subprocess.run(["sbatch", os.path.join(jobpath, "./jobscript.sh")])
    else:
        subprocess.run(["qsub", os.path.join(jobpath, "./jobscript.sh")])
