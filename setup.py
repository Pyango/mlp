from setuptools import setup
from Cython.Build import cythonize

setup(
    name='A neat app ;)',
    ext_modules=cythonize(["./entities/*.py", "./tradingpretrain.py"]),
    zip_safe=False,
)
