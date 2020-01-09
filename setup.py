from setuptools import setup, find_packages

setup(name='fitkit',
      version='0.0.1',
      description='Python methods and objects for fitting functions to data',
      url='https://github.com/Emigon/fitkit',
      author='Daniel Parker',
      author_email='danielparker@live.com.au',
      packages=find_packages(),
      install_requires=[
          'pint>=0.9',
          'sympy>=1.4',
          'numpy>=1.16.3',
          'scipy>=1.3.0',
          'pandas>=0.24.2',
          'matplotlib>=3.0.3',
        ])
