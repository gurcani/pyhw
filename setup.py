#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:46:13 2020

@author: ogurcan
"""

import subprocess
from setuptools import setup
#from distutils.command.install import install as _install
from distutils.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Specialized Python source builder."""
    subprocess.call(['make', 'clean', '-C', 'src/fhwak'])
    subprocess.call(['make', '-C', 'src/fhwak'])
#    _build_py.run(self)

setup(
    name='pyhw',
    version='0.0.1',
    author='Ozgur D. Gurcan',
    package_dir={'pyhw': 'src','pyhw.fhwak':'src/fhwak'},
    packages=['pyhw','pyhw.fhwak'],
    package_data={'pyhw.fhwak': ['libfhwak.so']},
    cmdclass={'build_py': build_py}
)
