import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='ForesightPy',
     packages = ['ForesightPy'],
     version='1.0.1',
     author="Oliver Boom",
     author_email="ob3618@ic.ac.uk",
     license='MIT',
     description="A package for price forecasting using LSTM networks",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/msc-acse/acse-9-independent-research-project-OliverJBoom",
     download_url = "https://github.com/msc-acse/acse-9-independent-research-project-OliverJBoom/archive/1.0.1.tar.gz",
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
