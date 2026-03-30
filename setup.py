from setuptools import setup, find_packages

setup(
    name='neuralflow-core',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['numpy', 'torch'],
    author='Alexander J. Thompson',
    description='High-performance neural network core library',
    url='https://github.com/Whoure2/NeuralFlow-Core',
)