from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fk_torchutils', 
    version='0.1.0',
    description='Some extensions/helpers for common functionality of PyTorch',  
    long_description=long_description, 
    long_description_content_type='text/markdown',
    url='https://github.com/fKunstner/torchutils',
    author='Frederik Kunstner',
    author_email='frederik.kunstner@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
		'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='torch pytorch torchutils',
    packages=find_packages(exclude=['contrib', 'docs', 'tests', 'other']),
    install_requires=['torch'],
    project_urls={
        'Documentation': 'https://fkunstner.github.io/torchutils/',
    },
)
