import shutil, os

def dt_str(dt):
    if dt >= 1.0:
        raise ValueError("Don't know how to represent {}".format(dt))
    return ("%.10f" % dt)[2:].rstrip('0')

for D in [32, 48, 64, 128]:
    for dt in [0.05, 0.02, 0.01, 0.005, 0.004, 0.003, 0.002, 0.001]:
        for suff in ["", "_RK4"]:
            found = []
            prefixes = [
                "expvals_evo",
                "schmidts_evo",
                "eta_sqs_evo",
                "proj_errs_evo",
                "truncerrs_lr",
                "truncerrs_rl"]

            for prefix in prefixes:
                try:
                    src_fn = "maxD{}_dt{}{}/{}.npy".format(D, dt_str(dt), suff, prefix)
                    targ_fn = "to_download/{}_D{}_dt{}{}.npy".format(prefix, D, dt_str(dt), suff)
                    os.stat(src_fn)
                    os.symlink("../" + src_fn, targ_fn)
                    found.append(prefix)
                except FileNotFoundError:
                    pass

            if found:
                print("D={}, dt={}{}: {}".format(D, dt, suff, found))
