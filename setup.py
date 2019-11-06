from setuptools import setup, find_packages

setup(name='dsptypes',
      version='0.0.1',
      description='Python classes for terse signal processing algorithm development and testing',
      url='https://github.com/Emigon/dsptypes',
      author='Daniel Parker',
      author_email='danielparker@live.com.au',
      packages=find_packages(),
      install_requires=[
          'pint>=0.9',
          'sympy>=1.4',
          'numpy>=1.16.3',
          'pandas>=0.24.2',
          'matplotlib>=3.0.3',
        ])
