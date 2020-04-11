
from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

setup(
    name='sns throughput prediction',
    version='0.1.0',
    description='SNS assignment o n throughput prediction',
    long_description=readme,
    author='Rashmi Patil',
    author_email='rashmipatil1492@gmail.com',
    url='https://github.com/rashmi-patil-1492/sns-throughput-prediction'
)