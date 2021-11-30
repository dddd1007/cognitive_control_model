# build the modules

from distutils.core import setup, Extension

setup(name='RLCC', version='1.0',  \
      ext_modules=[Extension('RLCC', ['RLCCmodule.c'])])