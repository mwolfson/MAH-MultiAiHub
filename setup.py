"""Module providing info about the package and its dependencies."""

from setuptools import setup, find_packages

setup(
    name='multiaihub',
    version='0.1.5',  # Start with a version number
    description='MAH makes it easy to send the same prompt to multiple LLMs',
    license='Apache License 2.0',
    long_description="MAH - Multi AI Hub is a project designed to make it easy to send the same prompt to multiple LLMs to help with testing and comparison.",
    author='Mike Wolfson',
    author_email='mike@ableandroid.com',
    packages=find_packages(),
    py_modules=["multi_ai_hub"],
    install_requires=[
        'python-dotenv==1.0.0',
        'google.generativeai==0.5.4',
        'openai==1.35.2',
        'anthropic==0.17.0',
    ],
)
