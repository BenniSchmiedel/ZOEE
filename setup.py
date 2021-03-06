import setuptools


#def readme():
#    with open('README.rst') as f:
#        return f.read()
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(name='ZOEE',
      version='0.1',
      description='A python implementation of Zero to One dimensional Earth Energy balance models',
      long_description=long_description,
      long_description_content_type='text/plain',
      url='https://github.com/BenniSchmiedel/ZOEE',
      author='Benjamin Schmiedel',
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=[
          'matplotlib',
          'numpy',
          'netCDF4',
          'tqdm',
          'qualname'
      ],
      classifiers=[
    'Development Status :: 3 - Alpha'],
      include_package_data=True,
      zip_safe=False)
