from setuptools import find_packages, setup

setup(
    name='oac',
    version='0.1.6',
    packages=find_packages(include=['optimizationalgorithmscomparison'], where='src'),
    description='Optimization algorithms comparison library',
    author='Andrei Telbukhov',
    lisence='',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas>=1.2.0'
    ],
    package_dir={
        '': 'src',
    }
)
