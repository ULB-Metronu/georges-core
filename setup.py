"""Installation with setuptools or pip."""
from setuptools import setup, find_packages
import os
import ast


def get_version_from_init():
    """Obtain library version from main init."""
    init_file = os.path.join(
        os.path.dirname(__file__), 'georges_core', '__init__.py'
    )
    with open(init_file) as fd:
        for line in fd:
            if line.startswith('__version__'):
                return ast.literal_eval(line.split('=', 1)[1].strip())


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    lic = f.read()


setup(
    name='georges_core',
    version=get_version_from_init(),
    description='Georges\' accelerator physics library - Core modules',
    long_description=readme,
    author='Cédric Hernaslteens',
    author_email='cedric.hernalsteens@ulb.be',
    url='https://github.com/ULB-Metronu/georges-core',
    license=lic,
    packages=find_packages(exclude=('tests', 'docs', 'examples')),
    install_requires=[
        'matplotlib',
        'numba',
        'numpy',
        'pandas',
        'pint',
        'plotly',
    ],
    package_data={'georges_core': []},
)
