from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt
def get_requirements():
    with open(Path(__file__).parent / "requirements.txt") as f:
        return f.read().splitlines()
    
setup(
    name="bayesian_dpddm",  # Package name
    version="1.0",  # Version
    author="viet",
    author_email="viet@cs.toronto.edu",
    description="Implements Bayesian D-PDDM.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/opent03/bayesian_dpddm",  # Project URL
    packages=find_packages(),  # Automatically find packages
    install_requires=get_requirements(),
    python_requires=">=3.11",  # Minimum Python version
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)