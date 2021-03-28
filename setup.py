from setuptools import find_packages, setup

setup(
    name='CutieFake',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/hashimom/CutieFake',
    license='MIT',
    author='Hashimoto Masahiko',
    author_email='hashimom@geeko.jp',
    entry_points={
        "console_scripts": [
            "cutiefake = cutiefake.server:main",
        ],
    },
    description=''
)
