from __future__ import print_function

import hashlib
import os
import subprocess
import time

import numpy as np


class GenerateStructure:
    def __init__(self, sig=6, two_d=False, lattice='lat.in'):
        self.sig = sig
        self.lattice = lattice
        self.two_d = two_d

    def generate(self, n=1):
        if self.two_d:
            two_d = "-2d"
        else:
            two_d = ""
        genstr = subprocess.check_output(["genstr", "-n=" + str(n),
                                          "-sig=" + str(self.sig),
                                          "-l=" + self.lattice,
                                          two_d])
        ostrs = genstr.split("end\n")
        for i, ostr in enumerate(ostrs):
            content = ostr.rstrip().lstrip()
            if content:
                ofile = open('str' + str(i) + '.out', 'w')
                ofile.write(content)
                ofile.close()

    def get_sig(self):
        return self.sig

    def get_lattice(self):
        return self.lattice

    def get_two_d(self):
        return self.two_d

    def set_sig(self, sig):
        self.sig = sig

    def set_lattice(self, lattice):
        self.lattice = lattice

    def set_two_d(self, two_d):
        self.two_d = two_d
