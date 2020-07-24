from setuptools import setup

setup(
    name='CutieFake',
    version='0.0.1',
    packages=['cutiefake', 'cutiefake.modelmaker', 'cutiefake.proto'],
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
        "marisa-trie",
        "transformers",
        "grpcio-tools"
    ],
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
