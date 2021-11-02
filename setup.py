from setuptools import setup, find_packages

setup(name='fitkit',
      version='0.4.0',
      description='Python methods and objects for fitting functions to data',
      url='https://github.com/Emigon/fitkit',
      author='Daniel Parker',
      author_email='danielparker@live.com.au',
      packages=find_packages(),
      install_requires=[
          'numpy>=1.21.2',
          'sympy>=1.8',
          'scipy>=1.7.1',
          'matplotlib>=3.4.3',
      ]
)
