from setuptools import find_packages, setup

setup(
    name='oac',
    version='0.1.2',
    packages=find_packages(include=['optimizationalgorithmscomparison'], where='src'),
    description='Optimization algorithms comparison library',
    author='Andrei Telbukhov',
    lisence='',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas'
    ],
    package_dir={
        '': 'src',
    }
)
