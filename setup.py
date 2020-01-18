from setuptools import setup

setup(
    name='CutieFake',
    version='0.0.1',
    packages=['cutiefake', 'cutiefake.modelmaker'],
    install_requires=[
        "tensorflow>=2.0.0",
        "numpy",
        "marisa-trie"
    ],
    url='https://github.com/hashimom/CutieFake',
    license='MIT',
    author='Hashimoto Masahiko',
    author_email='hashimom@geeko.jp',
    entry_points={
        "console_scripts": [
            "cutiefake = cutiefake.converter:main",
        ],
    },
    description=''
)
