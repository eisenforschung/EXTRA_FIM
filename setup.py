
from setuptools import setup, find_packages

setup(
    author="Shalini Bhatt",
    author_email='s.bhatt@mpie.de',
    python_requires='>=3.8',
    
    description="FIM study with EXTRA",
    install_requires = ['numpy', 'matplotlib', 'h5py', 'scipy',
    'netCDF4'],
            
    include_package_data=True,
    keywords='EXTRA_FIM',
    name='EXTRA_FIM',
    packages=find_packages(include=['EXTRA_FIM', 'EXTRA_FIM.*']),
)


