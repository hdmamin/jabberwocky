"""Py2app configuration. I keep this separate from the project root because
otherwise my ReadmeUpdater malfunctions and places notebook summaries in the
root README.
"""

from setuptools import setup
import sys


sys.setrecursionlimit(2_500)
OPTIONS = {
    'argv_emulation': False,
    'iconfile': '../data/icons/icon.icns',
    'plist': {
        'PyRuntimeLocations':
        ['@executable_path/../Frameworks/libpython3.7m.dylib',
         '/Users/hmamin/anaconda3/lib/libpython3.7m.dylib']
    }
}

setup(
    app=['../gui/main.py'],
    data_files=['../data'],
    setup_requires=['py2app'],
    options={'py2app': OPTIONS}
)
