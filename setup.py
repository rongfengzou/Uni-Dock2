"""
Uni-Dock Project
The Algorithm Package of DP Docking Platform
"""
import sys
from setuptools import setup, find_packages
from glob import glob
import versioneer

short_description = __doc__.split('\n')

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open('README.md', 'r') as handle:
        long_description = handle.read()
except:
    long_description = '\n'.join(short_description[2:])

install_requires = []

version_str = versioneer.get_version()
if version_str.startswith("base/"):
    version_str = version_str.partition(".")[2] + ".base"
setup(
    name='unidock2',
    author='DP Uni-Dock Team',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=version_str,
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',
    packages=find_packages(),
    install_requires=install_requires,
    data_files=[
        ('torsion_library', ['unidock/unidock_processing/torsion_library/data/torsion_library_2020.xml',
                             'unidock/unidock_processing/torsion_library/data/torsion_library_CDPKit.xml']),
        ('unidock_template', ['unidock/unidock_processing/unidocktools/data/unidock_option_template.yaml',
                              'unidock/unidock_processing/unidocktools/data/tleap_receptor_template.in']),
        ('bin', ['unidock/unidock_engine/build/bin/ud2'])
                             
    ],
    entry_points={},
    include_package_data=True,
    setup_requires=[] + pytest_runner
)
