import os
import setuptools


def requirements(path='requirements.txt'):
    with open(path, 'r') as f:
        deps = [line.strip() for line in f]
    return deps


def version():
    path = os.path.join('jabberwocky', '__init__.py')
    with open(path, 'r') as f:
        for row in f:
            if not row.startswith('__version__'):
                continue
            return row.split(' = ')[-1].strip('\n').strip("'")


setuptools.setup(name='jabberwocky',
                 version=version(),
                 author='Harrison Mamin',
                 author_email='harrisonmamin@gmail.com',
                 description='Core library powering a GUI providing an audio '
                             'interface to GPT3.',
                 install_requires=requirements(),
                 packages=setuptools.find_packages())

