from setuptools import find_packages, setup

setup(
    name='oac',
    version='0.1.1',
    packages=find_packages(include=['optimizationalgorithmscomparison']),
    description='Optimization algorithms comparison library',
    author='Andrei Telbukhov',
    lisence='',
    install_requires=['numpy']
)