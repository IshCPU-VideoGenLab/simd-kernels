from setuptools import setup, find_packages
setup(name="simd-kernels", version="0.1.0", author="Ishmael Affum Kwakye",
      description="Portable SIMD kernels for binary neural network inference",
      package_dir={"": "src"}, packages=find_packages(where="src"),
      python_requires=">=3.9", install_requires=["numpy>=1.24.0"])
