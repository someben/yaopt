#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = [
        'numpy',
        'scipy',
]

test_requirements = [
]

setup(
    name='yaopt',
    version='0.1.0',
    description='Basic options pricing in Python',
    long_description=readme + '\n\n' + history,
    author='Ben Gimpert',
    author_email='ben@somethingmodern.com',
    url='https://github.com/someben/yaopt',
    packages=[
        'yaopt',
    ],
    package_dir={'yaopt':
                 'yaopt'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='yaopt',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
