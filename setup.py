from setuptools import setup, find_packages

setup(name="trajectory-planner",
      package_dir={"": "src"},
      packages=find_packages(where="src"))
