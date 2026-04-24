from setuptools import setup, find_packages

setup(
    name="ur5e_env",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ur-rtde",          # pip install ur-rtde
        "numpy",
        "scipy",
        "gymnasium",
        "opencv-python",
        "pynput",
    ],
)
