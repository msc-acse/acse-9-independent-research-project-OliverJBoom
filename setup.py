import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='Foresight',
     version='1.0.1',
     author="Oliver Boom",
     author_email="ob3618@ic.ac.uk",
     license='MIT',
     description="A package for price forecasting using LSTM networks",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/msc-acse/acse-9-independent-research-project-OliverJBoom",
     download_url = "https://github.com/msc-acse/acse-9-independent-research-project-OliverJBoom/archive/1.0.1.tar.gz",
     packages=setuptools.find_packages(),
     install_requires=[
                "numpy>=1.16",
                "pandas>=0.24",
                "pmdarima>=1.2",
                "matplotlib>=3.0",
                "psycopg2>=2.7.6",
                "pytest",
                "scikit-learn>=0.20",
                "statsmodels>=0.9",
                "torch>=1.1",
      ],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
