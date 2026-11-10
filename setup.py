from setuptools import setup, find_packages

setup(
    name="sirc_mx",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(),
    # tools/dof_edge_kpi.py only; the library itself does not need OpenCV.
    extras_require={"eval": ["opencv-python-headless>=4.5"]},
)
