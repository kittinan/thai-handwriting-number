'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='thai-handwriting-number',
      version='0.1',
      packages=find_packages(),
      include_package_data=True,
      description='Thai Handwriting Number Keras Model on Cloud ML Engine',
      author='Kittinan',
      license='MIT',
      install_requires=[
          'keras',
          'h5py'],
      zip_safe=False)