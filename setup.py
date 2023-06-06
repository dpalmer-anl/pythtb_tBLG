# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:21:52 2023

@author: danpa
"""


from setuptools import setup, find_packages

setup(
        name='pythtbtBLG',
        version='0.0.1',
        author="Dan Palmer",
        author_email="dpalmer3@illinois.edu",
        description="python tight binding models for twisted bilayer graphene",
        url="https://github.com/dpalmer-anl/pythtb_tBLG",
        packages=['pythtbtBLG'],
        include_package_data=True,
        zip_safe=False,
        python_requires=">=3.6, <4",
        install_requires = ["numpy", "scipy", "pandas", "h5py", "ase", "pythtb",
                            'numba'],
)
