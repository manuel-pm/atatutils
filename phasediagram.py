from __future__ import print_function

import multiprocessing
try:
    import cPickle as pickle
except:
    import pickle
import os
import shutil
import subprocess
import time
import warnings

import numpy as np

from atatutils.clusterexpansion import ClusterExpansion as CE


def load_phase_diagram(filename):
    f = open(filename, 'rb')
    pd = pickle.load(f)
    f.close()
    return pd


def linear_interp(xp, yp):
    def monomial_basis(x):
        return np.interp(x, xp, yp)
    return monomial_basis


class PhaseDiagram:
    def __init__(self, cluster_expansion=None, pd_dir=None):
        self.has_cluster_expansion = False
        self.set_cluster_expansion(cluster_expansion)
        if pd_dir is None:
            self.pd_dir = 'phase_diagrams'
        else:
            self.pd_dir = pd_dir
        if not os.path.exists(self.pd_dir):
            os.mkdir(pd_dir)
        self.level = 0
        self.interpolate = []
        self.ECI = None
        self.prng_states = []
        self.n_eci_written = 0
        self.n_eci_run = multiprocessing.Value('i', 0)
        self.n_ecis = 0

    def set_cluster_expansion(self, cluster_expansion):
        self.ce = cluster_expansion
        if cluster_expansion is not None:
            self.has_cluster_expansion = True

    def generate_random_ECI(self, N=1):
        if not self.has_cluster_expansion:
            warnings.warn("Cluster Expansion must be set to generate ECI's")
            return
        ECI, prng_states = self.ce.sample_eci_with_state(N)
        if self.ECI is not None:
            self.ECI = np.vstack((self.ECI, ECI))
        else:
            self.ECI = ECI
        self.prng_states += prng_states
        self.n_ecis += N

    def write_eci_to_file(self, N=1, which=None):
        if which is None:
            which = self.n_eci_written
        if which + N > self.n_ecis:
            self.generate_random_ECI(N=which + N - self.n_ecis)
        for i in range(N):
            idx = which + i
            rank = self.prng_states[idx]
            np.savetxt('reci_' + str(rank) + '.out', self.ECI[idx, :])
            gsl, gsl_fold = self.ce.get_ground_state_line(ECI=self.ECI[idx, :])
            gsl_file = open('gs_str_' + str(rank) + '.out', 'w')
            for f in gsl_fold:
                tmp = open(os.path.join(f, 'str.out'), 'r')
                gsl_file.write(tmp.read())
                gsl_file.write('end\n\n')
                tmp.close()

        self.n_eci_written += N

    def _run_emc2(self, rank, start_phase, T0, T1, mu0, mu1, dT=15, dmu=0.04,
                  dx=1.e-2, er=50, k=8.617e-5, timeout=None):
        ofile = os.path.join(self.pd_dir, "emc2" + str(start_phase) +
                             "_" + str(rank) + ".out")

        process = subprocess.Popen(["emc2",
                                    "-gs=" + str(start_phase),
                                    "-T0=" + str(T0), "-T1=" + str(T1),
                                    "-dT=" + str(dT),
                                    "-mu0=" + str(mu0), "-mu1=" + str(mu1),
                                    "-dmu=" + str(dmu),
                                    "-dx=" + str(dx),
                                    "-er=" + str(er), "-k=" + str(k),
                                    "-o=" + ofile])

        print("PID:", process.pid)
        start = time.time()
        if timeout is None:
            timeout = 3600*24*10
        while time.time() - start <= timeout:
            if process.poll() is not None:
                break
            else:
                time.sleep(600)
        else:
            print("timed out, killing process")
            process.terminate()
        self.n_eci_run.value += 1

    def run_emc2(self, nproc, start_phase, T0, T1, mu0, mu1, dT=15, dmu=0.04,
                 dx=1.e-2, er=50, k=8.617e-5, timeout=None):
        self.write_eci_to_file(nproc)
        eci_run = self.n_eci_run.value
        lock = multiprocessing.Lock()
        for i in range(nproc):
            shutil.copy('gs_str_{}.out'.format(self.prng_states[eci_run + i]),
                        'gs_str.out')
            shutil.copy('reci_{}.out'.format(self.prng_states[eci_run + i]),
                        'eci.out')
        p = [multiprocessing.Process(target=self._run_emc2,
                                     args=(self.prng_states[eci_run + i],
                                           start_phase, T0, T1, mu0, mu1,
                                           dT, dmu, dx, er, k, timeout))
             for i in range(nproc)]
        for i in p:
            i.start()
        # self.wait_for_timeout(p, timeout)
        for i in p:
            i.join()

    def _run_phb(self, rank, phase_1, phase_2, timeout=None,
                 T0=None, mu0=None, dT=15, dx=1.e-2,
                 er=50, k=8.617e-5, ltep=1):
        ofile = os.path.join(self.pd_dir, "ph" + str(phase_1) + str(phase_2) +
                             "_" + str(self.level) + "_" + str(rank) + ".out")
        if T0 is not None:
            assert mu0 is not None
            process = subprocess.Popen(["phb",
                                        "-T=" + str(T0), "-mu=" + str(mu0),
                                        "-gs1=" + str(phase_1),
                                        "-gs2=" + str(phase_2),
                                        "-dT=" + str(dT), "-dx=" + str(dx),
                                        "-er=" + str(er), "-k=" + str(k),
                                        "-ltep=" + str(ltep),
                                        "-ecifile=" + "reci_" + str(rank) +
                                        ".out",
                                        "-o=" + ofile])
        else:
            assert mu0 is None
            process = subprocess.Popen(["phb",
                                        "-gs1=" + str(phase_1),
                                        "-gs2=" + str(phase_2),
                                        "-dT=" + str(dT), "-dx=" + str(dx),
                                        "-er=" + str(er), "-k=" + str(k),
                                        "-ltep=" + str(ltep),
                                        "-ecifile=" + "reci_" + str(rank) +
                                        ".out",
                                        "-o=" + ofile])
        print("PID:", process.pid)
        start = time.time()
        if timeout is None:
            timeout = 7200
        while time.time() - start <= timeout:
            if process.poll() is not None:
                break
            else:
                time.sleep(600)
        else:
            print("timed out, killing process")
            process.terminate()
        self.n_eci_run.value += 1

    def run_phb(self, phase_1, phase_2, nproc=1, timeout=7200,
                T0=None, mu0=None, dT=25, dx=1.e-2, er=50, k=8.617e-5,
                ltep=100):
        self.write_eci_to_file(nproc)
        eci_run = self.n_eci_run.value
        lock = multiprocessing.Lock()
        for i in range(nproc):
            shutil.copy('gs_str_' +
                        str(self.prng_states[eci_run + i]) +
                        '.out', 'gs_str.out')
        p = [multiprocessing.Process(target=self._run_phb,
                                     args=(self.prng_states[eci_run + i],
                                           phase_1, phase_2, timeout,
                                           T0, mu0, dT, dx, er, k, ltep))
             for i in range(nproc)]
        for i in p:
            i.start()
        # self.wait_for_timeout(p, timeout)
        for i in p:
            i.join()
        while True and (self.level < 4):
            refine_run = []
            for prng_id in self.prng_states[eci_run: eci_run + nproc]:
                try:
                    pdd = np.loadtxt(os.path.join(self.pd_dir,
                                                  "ph" + str(phase_1) +
                                                  str(phase_2) +
                                                  "_" + str(self.level) + "_" +
                                                  str(prng_id) + ".out"))
                except:
                    continue
                # mus = (pdd[:, 2] + 1)/2
                mus = pdd[:, 1]
                refine = np.where(np.abs(mus[1:] - mus[:-1]) > 1e-6)[0]
                if len(refine) > 0:
                    refine_from = refine[0]
                    refine_run.append((prng_id, pdd[refine_from, 0],
                                       pdd[refine_from, 1]))
                else:
                    refine_run.append((prng_id, pdd[-1, 0],
                                       pdd[-1, 1]))
            if len(refine_run) > 0:
                self.level += 1
                dT = dT/2
                p = [multiprocessing.Process(target=self._run_phb,
                                             args=(refine_run[i][0],
                                                   phase_1, phase_2, timeout,
                                                   refine_run[i][1],
                                                   refine_run[i][2],
                                                   dT, dx, er, k, ltep))
                     for i in range(len(refine_run))]
                for i in p:
                    i.start()
                # self.wait_for_timeout(p, timeout)
                for i in p:
                    i.join()
            else:
                break

    def wait_for_timeout(self, processes, timeout=7200):  # 42300):
        bool_list = [True]*len(processes)
        start = time.time()
        while time.time() - start <= timeout:
            for i, p in enumerate(processes):
                bool_list[i] = p.is_alive()

            print(bool_list)

            if np.any(bool_list):
                time.sleep(10)
            else:
                break
        else:
            print("timed out, killing all processes")
            for p in processes:
                p.terminate()

    def parse_current(self):
        idir = os.getcwd()
        os.chdir(self.pd_dir)
        vnames = [i[:-4] for i in os.listdir('.') if
                  os.path.isfile(os.path.join('.', i)) and
                  i.startswith('ph') and i.endswith('.out')]
        prng_ids = [i.split('_')[-1] for i in vnames]
        prng_ids = list(set(prng_ids))
        fnames = [i for i in os.listdir('.') if
                  os.path.isfile(os.path.join('.', i)) and
                  i.startswith('ph') and i.endswith('.out')]
        fnames_id = []
        for i in prng_ids:
            fnames_id.append([])
            for j, n in enumerate(vnames):
                if n.split('_')[-1] == i:
                    fnames_id[-1].append(fnames[j])

        self.vars = {}
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for i, f_id in enumerate(fnames_id):
                f_id.sort(key=lambda x: x.split('_')[1])
                d = []
                for j, f in enumerate(f_id):
                    try:
                        farray = np.loadtxt(f)
                        d.append(farray)
                        if len(d[-1].shape) == 1:
                            d[-1] = d[-1].reshape(1, d[-1].shape[0])
                    except Warning:
                        print("Warning:", f, "empty")
                        continue
                for idx in range(len(d)-1):
                    d[idx] = d[idx][d[idx][:, 0] < d[idx + 1][0, 0]]
                self.vars[f_id[0][:-4]] = np.concatenate(d)

        os.chdir(idir)

    def save_pickled_state(self, filename=None):
        if filename is None:
            filename = 'phase_diagram.pkl'
        f = open(filename, 'wb')
        pickle.dump(self, f)
        f.close()

    @classmethod
    def load_pickled_state(cls, filename):
        f = open(filename, 'rb')
        pd = pickle.load(f)
        f.close()
        return pd

    def fit_phase_diagram(self, interp='spline', degree=3):
        if interp == 'spline':
            try:
                from scipy.interpolate import interp1d
            except:
                print("WARNING: scipy not available. Using linear.")
                interp = 'linear'
                degree = 1

        self.parse_current()

        for k in self.vars.keys():
            T = self.vars[k][:, 0]
            xl = self.vars[k][:, 2]/2 + 0.5
            xr = self.vars[k][:, 3]/2 + 0.5
            x = np.concatenate((xl, xr))
            y = np.concatenate((T, T))
            p = x.argsort()
            x = x[p]
            y = y[p]
            if (interp == 'spline' or interp == 'polynomial') and degree > 1:
                x = x[3:-3]
                y = y[3:-3]
                if interp == 'spline':
                    self.interpolate.append(interp1d(x, y, kind=3))
                if interp == 'polynomial':
                    z = np.polyfit(x, y, deg=degree)
                    self.interpolate.append(np.poly1d(z))
            elif interp == 'linear' or degree == 1:
                self.interpolate.append(linear_interp(x, y))

    def plot(self, interp='spline', degree=3):
        import matplotlib.pyplot as plt
        if interp == 'spline':
            try:
                from scipy.interpolate import interp1d
            except:
                print("WARNING: scipy not available. Using linear.")
                interp = 'linear'
                degree = 1

        self.parse_current()

        all_y = np.empty((len(self.vars.keys()), 200))
        for i, k in enumerate(self.vars.keys()):
            is_atat = False
            T = self.vars[k][:, 0]
            xl = self.vars[k][:, 2]/2 + 0.5
            xr = self.vars[k][:, 3]/2 + 0.5
            x = np.concatenate((xl, xr))
            y = np.concatenate((T, T))
            p = x.argsort()
            x = x[p]
            y = y[p]
            if k.split('_')[1] == 'atat':
                is_atat = True
                c = 'k'
            else:
                c = '0.75'

            if (interp == 'spline' or interp == 'polynomial') and degree > 1:
                x = x[3:-3]
                y = y[3:-3]
                if interp == 'spline':
                    interpolate = interp1d(x, y, kind=3)
                if interp == 'polynomial':
                    z = np.polyfit(x, y, deg=degree)
                    interpolate = np.poly1d(z)
                xi = np.linspace(np.min(x), np.max(x), 200)
                yi = interpolate(xi)
                all_y[i, :] = yi
                lw = 1
                s = 30
                if is_atat:
                    lw = 2
                    s = 60
                plt.scatter(x, y, s=s, c=c, marker='o')
                plt.plot(xi, yi, color=c, lw=lw)
            elif interp == 'linear' or degree == 1:
                plt.plot(x, y, 'o-', color=c)
                interpolate = linear_interp(x, y)
                xi = np.linspace(np.min(x), np.max(x), 200)
                yi = interpolate(xi)
                all_y[i, :] = yi

        # if (interp == 'spline' or interp == 'polynomial') and degree > 1:
        y = np.percentile(all_y, [2.5, 50, 97.5], axis=0)
        plt.plot(xi, y[1, :], color='r', lw=2)
        plt.fill_between(xi, y[0, :], y[2, :], facecolor='red', alpha=0.2)

        plt.grid()
        plt.xlim([0, 1])
        plt.ylim(ymin=0)
        plt.xlabel("%" + "Ge" + " in " + "SiGe", fontsize=34)
        plt.ylabel("Temperature [K]", fontsize=34)
        plt.tick_params(axis='both', labelsize=30)
        plt.show(block=False)

if __name__ == "__main__":
    pd = PhaseDiagram()
    pd.plot()
    cluster_diameters = [9.0, 0.0, 0.0, 0.0, 0.0]

    ce = CE(diameters=cluster_diameters)
    ce.fit()
    pd.set_cluster_expansion(ce)
    pd.run_phb(0, 1)
