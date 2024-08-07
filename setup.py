from setuptools import find_packages, setup

REQUIRES = """
scikit-learn
ruamel.yaml
chardet
requests
quandl
cvxpy
cvxpylayers
gurobipy
torch
numpy
pandas
ipympl
basemap
Pillow
ortools
elkai
ruamel.yaml==0.17.21
"""


def get_install_requires():
    # with open("requirements.txt", encoding="utf-8") as f:
    #     REQUIRES = f.read()

    reqs = [req for req in REQUIRES.split("\n") if len(req) > 0]
    return reqs


with open("README.md", encoding="utf-8") as f:
    readme = f.read()


def do_setup():
    setup(
        # includes all other files
        # include_package_data=True,
        # package name
        name="openpto",
        # version
        version="0.1",
        description="",
        long_description=readme,
        long_description_content_type="text/markdown",
        install_requires=get_install_requires(),
        python_requires=">=3.7.0",
        # dependencies
        packages=find_packages(),
        keywords=["AI", "CO"],
        # Github repo
        url="",
        # author name
        author="Haoyu Geng",
        # mail address
        author_email="genghaoyu98@sjtu.edu.cn",
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
        ],
    )


if __name__ == "__main__":
    do_setup()
    # setup()
