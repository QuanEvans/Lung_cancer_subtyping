from setuptools import setup

setup(
    name='GSEs', # Name of the package
    version='1.0.0', # version of the package
    description='GSE tools', #  description of the package
    author='bioinf575Group', # author of the package
    install_requires=['GEOparse','sklearn','pandas','matplotlib','seaborn'], # list of all dependencies to be installed
)