from setuptools import setup, Extension, find_packages
import pybind11
import os

ext_modules = [
    Extension(
        "utils.rqa_utils_cpp",  # Note the module name is now under the "utils" package.
        [os.path.join("utils", "rqa_utils.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++14"]  # or "-std=c++17"
    )
]

setup(
    name='rqa_analysis',
    version='1.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
    ],
    ext_modules=ext_modules,
    description='A package for Recurrence Quantification Analysis (RQA)',
    author='Mike Richardson and Cathy Macpherson',
    author_email='michael.j.richardson@mq.edu.au',
    url='https://github.com/xkiwilabs/Recurrence-Quantification-Analysis',
)
