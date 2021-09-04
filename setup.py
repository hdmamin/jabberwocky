from setuptools import setup
import sys


sys.setrecursionlimit(2_500)
OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'data/icons/icon.icns',
    'plist': {
        'PyRuntimeLocations':
        ['@executable_path/../Frameworks/libpython3.7m.dylib',
         '/Users/hmamin/anaconda3/lib/libpython3.7m.dylib']
    }
}

setup(
    app=['bin/main.py'],
    data_files=['data'],
    setup_requires=['py2app'],
    options={'py2app': OPTIONS}
)
