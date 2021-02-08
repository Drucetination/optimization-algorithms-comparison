from setuptools import find_packages, setup

pkg_name = 'optimizationalgorithmscomparison'

setup(
    name=pkg_name,
    version='0.1.0',
    packages=find_packages(include=[pkg_name], where='src'),
    description='Optimization algorithms comparison library',
    author='Andrei Telbukhov',
    lisence='',
    python_requires='>=3.6',
    install_requires=[
    	'numpy',
    ],
    package_dir={
        '': 'src',
    }
)
