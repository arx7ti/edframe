from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension("fitps", ["data/highfreq/core/bindings.cpp"],
              include_dirs=[pybind11.get_include(), "cpp"],
              language="c++",
              extra_compile_args=["-std=c++11"])
]

setup(
    name="fitps",
    version="0.1",
    ext_modules=ext_modules,
)
