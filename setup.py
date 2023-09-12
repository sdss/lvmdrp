# encoding: utf-8
#
# setup.py

from setuptools import setup, find_packages


NAME = "lvmdrp"
VERSION = "0.1.1dev"


def run():
    setup(
        name=NAME,
        version=VERSION,
        license="BSD3",
        description="SDSSV-LVM Data Reduction Pipeline",
        long_description=open("README.rst").read(),
        author="Alfredo Mejia-Narvaez, Eric Pellegrini",
        author_email="alfredoj.32@gmail.com, ericpellegrini@outlook.com",
        keywords="astronomy software",
        url="https://github.com/sdss/lvmdrp",
        include_package_data=True,
        python_requires=">=3.8",
        install_requires=[
            "numpy>=1.24.1",
            "scipy>=1.10.1",
            "matplotlib>=3.6.2",
            "pandas>=1.5.3",
            "tqdm>=4.64.1",
            "astropy>=5.2",
            "ccdproc>=2.4.0",
            "dust_extinction>=1.1",
            "skyfield>=1.45",
            "pydl>=0.7.0",
            "skycalc_cli==1.4",
            "sdsstools>=1.0",
            "sdss-tree>=4.0",
            "sdss-access>=3.0",
            "click>=8.0",
            "cloup>=2.0",
            "bottleneck>=1.3.7",
            "h5py>=3.8.0",
            "numexpr==2.8.4"
        ],
        extras_require={
            "dev": [
                "ipython>=8.10.0",
                "jupyter>=1.0.0",
                "jupyterlab>=3.6.1",
                "ipywidgets>=8.1.0",
                "ipympl==0.9.3"
            ]
        },
        packages=find_packages(where="python"),
        package_dir={"": "python"},
        scripts=["bin/drp", "bin/pix2wave", "bin/envcheck"],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.8",
            "Topic :: Documentation :: Sphinx",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],
    )

if __name__ == "__main__":
    run()
