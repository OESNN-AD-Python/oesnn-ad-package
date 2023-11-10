from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.9.12'
DESCRIPTION = 'OeSNN anomaly detector implementation for Python.'
LONG_DESCRIPTION = 'Online evolutionary Spiking Neural Network anomaly detector implementation for Python.'

# Setting up
setup(
    name="OeSNN-AD",
    version=VERSION,
    author="Mariusz Paluch",
    author_email="mariuszpaluch001@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy'],
    keywords=['python', 'data stream', 'spiking neural network', 'SNN', 'anomalies', 'anomaly detection'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)