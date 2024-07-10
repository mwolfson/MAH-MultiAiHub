"""Module providing info about the package and its dependencies."""

from setuptools import setup, find_packages

setup(
    name='multiaihub',
    version='0.1.0',  # Start with a version number
    description='Short description of your project',
    author='Mike Wolfson',
    author_email='mike@ableandroid.com',
    packages=find_packages(),
    install_requires=[
        'python-dotenv',
        'google.generativeai',
        'openai',
        'anthropic',
    ],
)
