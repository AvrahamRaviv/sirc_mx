from setuptools import setup, find_packages

setup(
    name="sirc_mx",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(),
)
