from setuptools import setup, find_packages
from typing import List

def get_requirements(filepath: str)->List[str]:
    HYPHEN_E_DOT = '-e .'
    with open('requirements.txt', 'r') as f:
        requirements = [req.replace('\n', '') for req in f if HYPHEN_E_DOT not in req]
    
    return requirements

setup(
    name="ML Project",
    version="0.0.1",
    author="Bijon Lahiri",
    author_email="bijonlahiri@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)