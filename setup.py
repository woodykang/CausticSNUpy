import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
 
setuptools.setup(
    name="causticsnupy",
    version="1.0.1",
    author="Wooseok Kang",
    author_email="woodykang@snu.ac.kr",
    description="Python package for determining cluster members using caustic technique.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/woodykang/CausticSNUpy",
    packages=setuptools.find_packages(),
    setup_requires = ['numpy>=1.20.3', 'scipy>=1.7.1', 'astropy>=4.3.1', 'scikit-image>=0.17.2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3",
        "Operating System :: OS Independent",
    ],
)