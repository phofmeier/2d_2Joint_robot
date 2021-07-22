from setuptools import setup, find_packages

install_requirements = ["Flask", "numpy", "casadi", "pandas", "tensorflow"]

setup(name="robot-animation",
      setup_requires=['setuptools_scm', 'setuptools'],
      use_scm_version=True,
      package_dir={"": "src"},
      packages=find_packages(where="src"),
      install_requires=install_requirements,
      author='Peter Hofmeier',
      author_email='phofmeier@gmail.com',
      entry_points={
          'console_scripts': ['robotAnimation=robot_animation.main:main']
      }
      )
