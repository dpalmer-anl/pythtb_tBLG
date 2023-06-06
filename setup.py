# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:21:52 2023

@author: danpa
"""


from setuptools import setup

MAIN_PACKAGE = 'pythtbtBLG'
DESCRIPTION = "Python tight binding models for twisted bilayer graphene"
LICENSE = "GPLv3"
URL = "https://github.com/dpalmer-anl/pythtb_tBLG"
AUTHOR = "Dan Palmer"
EMAIL = "dpalmer3@illinois.edu"
VERSION = '0.0.1'
CLASSIFIERS = [
               'Environment :: Console',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
               'Natural Language :: English',
               'Programming Language :: Python',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 3',
               'Topic :: Scientific/Engineering :: Chemistry',
               'Topic :: Scientific/Engineering :: Physics']
DEPENDENCIES = ["numpy", "scipy", "pandas", "h5py", "ase", "pythtb",'numba']
KEYWORDS = 'physics quantum mechanics solid state'


def readme():
    'Return the contents of the README.md file.'
    with open('README.rst') as freadme:
        return freadme.read()


def setup_package():

    setup(name=MAIN_PACKAGE,
          version=VERSION,
          url=URL,
          description=DESCRIPTION,
          author=AUTHOR,
          author_email=EMAIL,
          license=LICENSE,
          keywords=KEYWORDS,
          long_description=readme(),
          classifiers=CLASSIFIERS,
          packages=['pythtbtBLG'],
          install_requires=DEPENDENCIES,
          )


if __name__ == "__main__":
    setup_package()