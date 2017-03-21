from __future__ import print_function

import copy
from fractions import Fraction
import ntpath
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import yaml

# from ase.calculators.calculator import Parameters

from atatutils.str2gpaw import read_lattice_file
from utils.bcolors import bcolors, print_error, print_success


def frac_to_float(fraction):
    """ Evaluates a fraction given as Fraction object or as a string into a
    float. If an integer is given, it is interpreted as 1/int """
    if isinstance(fraction, int):
        return 1. / float(fraction)
    if isinstance(fraction, Fraction):
        return float(fraction)
    data = fraction.strip().split('/')
    num = float(data[0])
    den = float(data[1])
    return num / den


class SMCJobDescriptor(object):
    """ Class to manage the inputs to MPDC """
    __default_parameters = {'n_particles': 1,
                            'n_mcmc_passes': 10,
                            'n_init_mcmc_passes': 400,
                            'ess_reduction': 0.95,
                            'ess_threshold': 0.67,
                            'output_prefix': 'smc_1/smc_x_1',
                            'verbosity_SMC': 1,
                            'T_i': 2000,
                            'T_f': 10,
                            'mu_i': 0.0,
                            'mu_f': 0.0,
                            'T_step': 10.,
                            'enclosing_radius': 50.,
                            'concentration': '0/1',
                            'initial_structure': '\"\"',
                            'canonical': 1,
                            'verbosity_MPDC': 1,
                            'output_frequency': 5000,
                            'initial_folder': 'empty',
                            'seed': 314159265}
    'Default parameters'

    def __init__(self, filename=None, parameters=None):
        self.__params = {}
        self.__params.update(self.__default_parameters)
        if filename is not None:
            self.update_parameters(filename)
        if parameters is not None and isinstance(parameters, dict):
            self.set_parameters(**parameters)

    @property
    def concentration(self):
        return self.__params['concentration']

    @concentration.setter
    def concentration(self, x):
        self.__params['concentration'] = x

    @property
    def mu(self):
        return [self.__params['mu_i'], self.__params['mu_f']]

    @mu.setter
    def mu(self, x):
        self.__params['mu_i'], self.__params['mu_f'] = x

    @property
    def enclosing_radius(self):
        return self.__params['enclosing_radius']

    @enclosing_radius.setter
    def enclosing_radius(self, er):
        self.__params['enclosing_radius'] = er

    @property
    def initial_folder(self):
        return self.__params['initial_folder']

    @initial_folder.setter
    def initial_folder(self, initial_folder):
        self.__params['initial_folder'] = initial_folder

    def is_canonical(self):
        return bool(self.__params['canonical'])

    def is_grand_canonical(self):
        return not bool(self.__params['canonical'])

    def get_default_parameters(self):
        return copy.deepcopy(self.__default_parameters)

    def get_parameters(self):
        return copy.deepcopy(self.__params)

    def set_parameters(self, **kwargs):
        """Set parameters like set_parameters(key1=value1, key2=value2, ...).

        A dictionary containing the parameters that have been changed
        is returned.

        The special keyword 'parameters' can be used to read
        parameters from a file."""

        if 'parameters' in kwargs:
            import yaml
            filename = kwargs.pop('parameters')
            parameters = yaml.load(open(filename))
            parameters.update(kwargs)
            kwargs = parameters

        changed_parameters = {}

        for key, value in kwargs.items():
            oldvalue = self.__params.get(key)
            if key not in self.__params or not (value == oldvalue):
                if isinstance(oldvalue, dict):
                    # Special treatment for dictionary parameters:
                    for name in value:
                        if name not in oldvalue:
                            raise KeyError(
                                'Unknown subparameter "{}" in '
                                'dictionary parameter "{}"'.format(name, key))
                    oldvalue.update(value)
                    value = oldvalue
                changed_parameters[key] = value
                self.__params[key] = value

        return changed_parameters

    def update_parameters(self, config):
        """ Updates the input parameters from a dictionary or a yaml file """
        if isinstance(config, str):
            with open(config) as ifile:
                parameters = yaml.load(ifile)
        elif isinstance(config, dict):
            parameters = config
        else:
            raise TypeError

        # filtered_parameters = {k: parameters[k] for k in self.__params}
        filtered_parameters = {k: parameters[k] for k in parameters.keys() if
                               k in self.__params}
        changed_parameters = {}

        for key, value in filtered_parameters.items():
            oldvalue = self.__params.get(key)
            if key not in self.__params or not (value == oldvalue):
                if isinstance(oldvalue, dict):
                    # Special treatment for dictionary parameters:
                    for name in value:
                        if name not in oldvalue:
                            print_error('Unknown subparameter "{}" in '
                                'dictionary parameter "{}"'.format(name, key))
                            raise KeyError
                    oldvalue.update(value)
                    value = oldvalue
                changed_parameters[key] = value
                self.__params[key] = value

        return changed_parameters

    def save_parameters(self, filename):
        """ Saves the parameters as a yaml file """
        import yaml
        d = yaml.dump(self.__params, default_flow_style=False)
        with open(filename, 'w+') as ofile:
            ofile.write(d)

    def write_input_file(self, filename):
        """ Saves the parameters as a MPDC input file """
        with open(filename, 'w+') as ofile:
            ofile.write('{} \\\n'.format(self.__params['n_particles']))
            ofile.write('{} \\\n'.format(self.__params['n_mcmc_passes']))
            ofile.write('{} \\\n'.format(self.__params['n_init_mcmc_passes']))
            ofile.write('{} \\\n'.format(self.__params['ess_reduction']))
            ofile.write('{} \\\n'.format(self.__params['ess_threshold']))
            ofile.write('{} \\\n'.format(self.__params['output_prefix']))
            ofile.write('{} \\\n'.format(self.__params['verbosity_SMC']))
            ofile.write('{} \\\n'.format(self.__params['T_i']))
            ofile.write('{} \\\n'.format(self.__params['T_f']))
            ofile.write('{} \\\n'.format(self.__params['mu_i']))
            ofile.write('{} \\\n'.format(self.__params['mu_f']))
            ofile.write('{} \\\n'.format(self.__params['T_step']))
            ofile.write('{} \\\n'.format(self.__params['enclosing_radius']))
            ofile.write('{} \\\n'.format(self.__params['concentration']))
            ofile.write('{} \\\n'.format(self.__params['initial_structure']))
            ofile.write('{} \\\n'.format(self.__params['canonical']))
            ofile.write('{} \\\n'.format(self.__params['verbosity_MPDC']))
            ofile.write('{} \\\n'.format(self.__params['output_frequency']))
            ofile.write('{} \\\n'.format(self.__params['initial_folder']))
            ofile.write('{}'.format(self.__params['seed']))


class SMCJob:
    """ Class to mange a single instance of an MPDC job """
    __default_parameters = {'base_folder': '.',
                            'ce_data_folder': '.',
                            'submission_folder': '.',
                            'clusters_filename': 'clusters.out',
                            'lattice_filename': 'lat.in',
                            'n_proc': 1}
    'Default parameters'

    def __init__(self, config=None):
        """ Initialise parameters from a config dictionary or yaml file """
        self.job_keys = ['base_folder', 'ce_data_folder', 'submission_folder',
                         'clusters_filename', 'lattice_filename',
                         'submission_command', 'n_proc', 'run_script']
        self.__params = {}
        self.__params.update(self.__default_parameters)
        self.job_descriptor = None
        if config is not None:
            self.update_parameters(config)
        self.support_files = ['ANALYZE_DATA.sh', 'EXTRACT_PAIR_CORRELATION.sh']
        self.__params['base_folder'] = os.path.abspath(
            self.__params['base_folder'])
        # self.set_descriptor(SMCJobDescriptor(config))

    def get_cluster_file(self):
        return os.path.join(self.__params['ce_data_folder'],
                            self.__params['clusters_filename'])

    def get_lattice_file(self):
        return os.path.join(self.__params['ce_data_folder'],
                            self.__params['lattice_filename'])

    def get_concentration(self, rtype=str):
        x = self.job_descriptor.concentration
        rx_frac = (Fraction(x) + 1) / 2
        return rtype(rx_frac)

    def get_supporting_files(self):
        return [os.path.join(self.__params['base_folder'], sf) for
                sf in self.support_files]

    def update_parameters(self, config):
        """ Updates the job parameters from a dictionary or yaml file """
        if isinstance(config, str):
            with open(config) as ifile:
                config = yaml.load(ifile)
        elif isinstance(config, dict):
            pass
        else:
            raise TypeError

        # filtered_config = {k: config[k] for k in self.job_keys}
        filtered_config = {k: config[k] for k in config.keys() if
                           k in self.job_keys}
        self.__params.update(filtered_config)
        self.__params['base_folder'] = os.path.abspath(
            self.__params['base_folder'])
        if self.job_descriptor is not None:
            self.update_descriptor(config)
        else:
            self.set_descriptor(SMCJobDescriptor(config))

    def update_enclosing_radius(self):
        """ Updates the enclosing radius so that the number of atoms is a
        multiple of the concentration (otherwise it ca not be represented
        exactly) """
        lcell, lpositions, latoms = read_lattice_file(
            self.__params['lattice_filename'])
        icell = np.linalg.inv(lcell)
        er = self.job_descriptor.enclosing_radius
        sim_size = np.ceil(np.linalg.norm(icell,
                                          axis=1) * 2 * er).astype(np.int)
        tot_atoms = sim_size[0] * sim_size[1] * sim_size[2]
        x_frac = self.job_descriptor.concentration.split('/')
        rx_frac = (Fraction(int(x_frac[0]), int(x_frac[1])) + 1) / 2
        rem = tot_atoms % rx_frac.denominator
        while (rem != 0):
            er += 1.
            sim_size = np.ceil(np.linalg.norm(icell,
                                              axis=1) * 2 * er).astype(np.int)
            tot_atoms = sim_size[0] * sim_size[1] * sim_size[2]
            # print("Increasing enclosing radius to {} for x={} "
            #       "(supercell atoms: {})".format(er, rx_frac, tot_atoms))
            rem = tot_atoms % rx_frac.denominator
        if er != self.job_descriptor.enclosing_radius:
            print("Enclosing radius for x={}: {} "
                  "(supercell atoms: {})".format(rx_frac, er, sim_size))
        self.job_descriptor.enclosing_radius = er

    def update_output_prefix(self, restart=False):
        """ Updates the prefix that will be appended to the folder of every
        simulation step """
        if self.job_descriptor.is_canonical():
            x = self.get_concentration(float)
            ppconc = '{:05}'.format(round(x * 100, 1))
        else:
            fconc = float(self.job_descriptor.mu[0])
            ppconc = '{:+06}'.format(fconc)
        if not restart:
            prefix = 'smc_{}'.format(self.__params['n_proc'])
        else:
            sdir = self.__params['submission_folder']
            rpattern = 'smc_{}_restart'.format(self.__params['n_proc'])
            restart_folders = [d for d in os.listdir(sdir)
                               if os.path.isdir(os.path.join(sdir, d))
                               and rpattern in d]
            if restart_folders:
                restart_folders.sort()
                nrestart = int(restart_folders[-1][-1]) + 1
            else:
                nrestart = 0

            prefix = ('smc_{0}_restart{1}/'.format(self.__params['n_proc'],
                                                   nrestart))
        if self.job_descriptor.is_canonical():
            prefix = os.path.join(prefix,
                                  'smc_{}perc_{}'.format(ppconc,
                                                         self.__params[
                                                             'n_proc']))
        else:
            prefix = os.path.join(prefix,
                                  'smc_mu{}_{}'.format(ppconc,
                                                       self.__params['n_proc']))
        self.update_descriptor({'output_prefix': prefix})

    def set_descriptor(self, job_descriptor):
        self.job_descriptor = job_descriptor
        self.update_enclosing_radius()
        self.update_output_prefix()

    def update_descriptor(self, config):
        """ Updates the job descriptor with the given dictionary or yaml
        file """
        self.job_descriptor.update_parameters(config)
        if isinstance(config, str):
            with open(config) as ifile:
                parameters = yaml.load(ifile)
        elif isinstance(config, dict):
            parameters = config
        else:
            raise TypeError

        if any(option in parameters for option in ['lattice_filename',
                                                   'concentration',
                                                   'enclosing_radius']):
            self.update_enclosing_radius()

        if any(option in parameters for option in ['concentration',
                                                   'n_proc', 'mu_i', 'mu_f']):
            self.update_output_prefix()

    def set_submission_command(self, command):
        self.__params['submission_command'] = command

    def prepare_submission(self):
        """ Creates the necessary directories and input files to run the job """
        nproc = self.__params['n_proc']
        sdir = self.__params['submission_folder']
        bdir = self.__params['base_folder']
        if not os.path.isdir(sdir):
            os.makedirs(sdir)
        shutil.copy(os.path.join(bdir, self.__params['clusters_filename']),
                    os.path.join(sdir, 'clusters.out'))
        shutil.copy(os.path.join(bdir, self.__params['lattice_filename']),
                    os.path.join(sdir, 'lat.in'))
        shutil.copy(os.path.join(bdir, self.__params['run_script']),
                    os.path.join(sdir, self.__params['run_script']))
        self.job_descriptor.write_input_file('ARGUMENTS_tmp')
        shutil.move('ARGUMENTS_tmp',
                    os.path.join(sdir, 'ARGUMENTS_{}'.format(nproc)))

    def submit(self, restart):
        """ Submits the job using the provided submission command """
        curdir = os.getcwd()
        try:
            os.chdir(self.__params['submission_folder'])
            subprocess.call(['{}'.format(self.__params['submission_command'])],
                            shell=True)
            if restart:
                open('restarted', 'a').close()
            os.chdir(curdir)
        except:
            print("Submission failed (command: {})".format(
                self.__params['submission_command']))
            os.chdir(curdir)

    def prepare_restart(self):
        """ Updates the input files for restarting the job """
        nproc = self.__params['n_proc']
        sdir = self.__params['submission_folder']
        bdir = self.__params['base_folder']
        self.update_output_prefix(restart=True)
        restart_filename = os.path.join(sdir, 'restart_filename.txt')
        initial_folder = open(restart_filename).readline()
        self.job_descriptor.initial_folder = initial_folder
        if not os.path.isfile(os.path.join(sdir,
                                           self.__params['run_script'])):
            shutil.copy(os.path.join(bdir, self.__params['run_script']),
                        os.path.join(sdir, self.__params['run_script']))
            # self.write_run_script(os.path.join(sdir,
            #                                    self.__params['run_script']))
        self.job_descriptor.write_input_file('ARGUMENTS_tmp')
        shutil.move('ARGUMENTS_tmp',
                    os.path.join(sdir, 'ARGUMENTS_{}_restart'.format(nproc)))

    def write_run_script(self, filename):
        """ Writes a script to submit the job """
        # TODO: redesign to allow for more flexibility in the queueing system
        # TODO: remove hard-coded references to 64
        with open(filename, 'w+') as ofile:
            ofile.write('#!/bin/bash\n')
            if self.__params['sub_script_type'] == 'pbs':
                ofile.write('#PBS -l nodes=4:ppn=16\n'
                            '#PBS -l pmem=3882mb\n'
                            '#PBS -l walltime=12:00:00\n'
                            '\n'
                            'module load intel'
                            'module load impi'
                            'module load imkl'.format())
            ofile.write('if [ ! -f ARGUMENTS_64 ]\n'
                        'then\n'
                        '    echo "ERROR: You need a file containing the arguments for the \n'
                        '    run called \\"ARGUMENTS_64\\""\n'
                        '    exit\n'
                        'fi\n'
                        '\n'
                        '#unlimit the stack (important, since default is 10 mb!):\n'
                        'ulimit -s unlimited\n'
                        'ulimit -u\n'
                        '\n'
                        '#print resource info\n'
                        'ulimit -a\n'
                        '#do we have required modules?\n'
                        '\n'
                        'echo "** submitting job"\n'
                        'date\n'
                        'path=$HOME/Programs/Jesper/PROGRAM/program\n'
                        '#NOTICE: Arguments for the run are in the file "ARGUMENTS"\n'
                        'cat $PWD/ARGUMENTS_64'.format())
            if self.__params['sub_script_type'] == 'pbs':
                ofile.write('srun $path/bin/MPDC_opt '
                            '`awk \'{printf $1 " "}\' '
                            '${PWD}/ARGUMENTS_64`\n'.format())
            else:
                ofile.write('mpirun -np 1 $path/bin/MPDC_debug '
                            '`awk \'{printf $1 " "}\' '
                            '${PWD}/ARGUMENTS_64`\n'.format())


class SMCJobHandler:
    """ Class to manage a collection of MPDC jobs """
    __default_parameters = {'n_samples': 1,
                            'n_proc': 1,
                            't_dep_eci': False,
                            'eci_folder': '\'../ecis\'',
                            'concentrations': ['0/1'],
                            'canonical': 1,
                            'mus': [0.0],
                            'sub_script_type': 'sh',
                            'sub_script_base_name': 'run_MPDC_',
                            'sub_script_id': 'n_proc',
                            'sub_command_pattern': ('./>sub_script< '
                                                    '| tee output.log')}
    'Default parameters'

    def __init__(self, config=None):
        """ Initialise parameters from a config dictionary or yaml file """
        self.job_keys = ['n_samples', 'n_proc', 't_dep_eci', 'eci_folder',
                         'concentrations', 'canonical', 'mus',
                         'sub_script_type', 'sub_command_pattern',
                         'sub_script_type', 'sub_script_base_name',
                         'sub_script_id', 'sub_script']
        self.__params = {}
        self.__params.update(self.__default_parameters)
        self.__params['sub_script'] = '{}{}.{}'.format(
            self.__params['sub_script_base_name'],
            self.__params[self.__params['sub_script_id']],
            self.__params['sub_script_type'])
        self.config = None
        if config is not None:
            self.config = config
            self.update_parameters(config)
        self.__params['eci_folder'] = os.path.abspath(
            self.__params['eci_folder'])
        self.n_jobs = (self.__params['n_samples'] *
                       len(self.__params['concentrations']))

    def update_parameters(self, config):
        """ Update parameters from a config dictionary or yaml file """
        if isinstance(config, str):
            with open(config) as ifile:
                config = yaml.load(ifile)
        elif isinstance(config, dict):
            pass
        else:
            raise TypeError

        filtered_config = {k: config[k] for k in config.keys() if
                           k in self.job_keys}
        self.__params.update(filtered_config)
        self.__params['eci_folder'] = os.path.abspath(
            self.__params['eci_folder'])
        self.n_jobs = (self.__params['n_samples'] *
                       len(self.__params['concentrations']))

        if any(option in config for option in ['sub_script_type',
                                               'sub_script_base_name',
                                               'sub_script_id']):
            self.__params['sub_script'] = '{}{}.{}'.format(
                self.__params['sub_script_base_name'],
                self.__params[self.__params['sub_script_id']],
                self.__params['sub_script_type'])

        self.__params['sub_command'] = self.__params['sub_command_pattern']
        for field in self.job_keys:
            self.__params['sub_command'] = re.sub('>' + field + '<',
                                                  str(self.__params[field]),
                                                  self.__params['sub_command'])

    def create_jobs(self, jobids=None):
        """ Create a collection of jobs.

         There are three main parameters defining the jobs:
          1.- Jobs are created from a randomly selected ECI file stored in a
          given
            directory ['eci_folder'].
          2.- The temperature range is used within each job, and a job...
           2a. - is created for each concentration (if canonical)
           2b. - is created for each chemical potential (if
           semi-grand-canonical)
        """
        self.jobs = []
        nsamples = self.__params['n_samples']
        nproc = self.__params['n_proc']
        tdep = self.__params['t_dep_eci']
        eci_f = self.__params['eci_folder']
        is_canonical = bool(self.__params['canonical'])
        if is_canonical:
            conc = self.__params['concentrations']
        else:
            conc = self.__params['mus']
        if isinstance(conc, str):
            conc = eval(conc)

        # Find available ECIs
        recis = [i[:-4] for i in os.listdir(eci_f) if
                 os.path.isfile(os.path.join(eci_f, i)) and i.endswith(
                     '.eci')]
        prefix = recis[0].split('_')[0]
        recis = [i[len(prefix) + 1:] for i in recis]
        recis.sort()

        # Choose ECIs randomly from unused ones or from the given set
        fname = "submitted_" + str(nproc) + ".dat"
        if jobids is None:
            self.submitted = []
            if os.path.isfile(fname):
                f = open(fname)
                for line in f:
                    self.submitted.append(line.rstrip())
                f.close()

            newrecis = recis
            for s in self.submitted:
                try:
                    newrecis.remove(s)
                except:
                    continue

            if len(newrecis) < nsamples:
                print("Only {} remaining ECIs available".format(len(newrecis)))
                nsamples = len(newrecis)
                params['n_samples'] = nsamples

            np.random.shuffle(newrecis)
            submissions = newrecis[: nsamples]
        else:
            if isinstance(jobids, list):
                submissions = jobids
            elif isinstance(jobids, str):
                submissions = [jobids]
            else:
                raise ValueError

        # Create the jobs to submit. This creates folders and input files
        f = open(fname, 'a')
        main_dir = os.getcwd()
        for s in submissions:
            print("Submitting in {}".format(s))
            for c in conc:
                job_params = {}
                if is_canonical:
                    fconc = round(100 * (frac_to_float(c) + 1) / 2, 1)
                    ppconc = '{:05}'.format(fconc)
                else:
                    fconc = float(c)
                    ppconc = '{:+06}'.format(fconc)
                if is_canonical:
                    submit_dir = os.path.abspath(
                        os.path.join(s, '{}perc'.format(ppconc)))
                else:
                    submit_dir = os.path.abspath(
                        os.path.join(s, 'mu{}'.format(ppconc)))
                job_params['submission_folder'] = submit_dir
                run_script = self.__params['sub_script']
                job_params['run_script'] = run_script
                job_params['id'] = s
                # job_params['n_proc'] = nproc
                if is_canonical:
                    job_params['concentration'] = c
                else:
                    job_params['mu_i'] = c
                    job_params['mu_f'] = c
                self.jobs.append(SMCJob(self.config))
                self.jobs[-1].update_parameters(job_params)
                self.jobs[-1].prepare_submission()
                if tdep:
                    eci_file = os.path.join(eci_f, '{}_{}.teci'.format(prefix,
                                                                       s))
                    trange_file = os.path.join(eci_f, 'Trange.in')
                    tecifile = open(os.path.join(submit_dir, 'teci.out'), 'w+')
                    trange = open(trange_file)
                    tecifile.write(trange.read())
                    trange.close()
                    teci_val = open(eci_file)
                    tecifile.write(teci_val.read())
                    teci_val.close()
                    tecifile.close()

                else:
                    eci_file = os.path.join(eci_f, '{}_{}.eci'.format(prefix,
                                                                      s))
                    shutil.copy(eci_file, os.path.join(submit_dir, 'eci.out'))
            f.write(s + '\n')
            os.chdir(main_dir)
        f.close()

    def submit_jobs(self, restart=False):
        """ Submit all the jobs in the handler """
        for job in self.jobs:
            job.set_submission_command(self.__params['sub_command'])
            job.submit(restart)

    def create_restart_jobs(self, restarts):
        """ Create a collections of jobs for restarting.

         Restart jobs always need to be given. It will update input files and
         create any necessary folder
        """
        self.jobs = []
        is_canonical = bool(self.__params['canonical'])
        if is_canonical:
            conc = self.__params['concentrations']
        else:
            conc = self.__params['mus']
        if isinstance(conc, str):
            conc = eval(conc)
        main_dir = os.getcwd()
        for s in restarts:
            for c in conc:
                job_params = {}
                if is_canonical:
                    fconc = round(100 * (frac_to_float(c) + 1) / 2, 1)
                    ppconc = '{:05}'.format(fconc)
                else:
                    fconc = float(c)
                    ppconc = '{:+06}'.format(fconc)
                if is_canonical:
                    restart_dir = os.path.abspath(
                        os.path.join(s, '{}perc'.format(ppconc)))
                else:
                    restart_dir = os.path.abspath(
                        os.path.join(s, 'mu{}'.format(ppconc)))
                if os.path.isfile(os.path.join(restart_dir, 'finished')):
                    print_success("Skipping {}: finished".format(restart_dir))
                    continue
                job_params['submission_folder'] = restart_dir
                run_script = self.__params['sub_script']
                job_params['run_script'] = run_script
                job_params['id'] = s
                # job_params['n_proc'] = nproc
                if is_canonical:
                    job_params['concentration'] = c
                else:
                    job_params['mu_i'] = c
                    job_params['mu_f'] = c
                self.jobs.append(SMCJob(self.config))
                self.jobs[-1].update_parameters(job_params)
                self.jobs[-1].prepare_restart()
            os.chdir(main_dir)
