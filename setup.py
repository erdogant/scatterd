import setuptools
import re

#-------- Version control
VERSIONFILE="scatterd/__init__.py"
getversion = re.search( r"^__version__ = ['\"]([^'\"]*)['\"]", open(VERSIONFILE, "rt").read(), re.M)
if getversion:
    new_version = getversion.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE))

#--------  Create setup file    
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     install_requires=['matplotlib','numpy','colourmap','seaborn'],
     python_requires='>=3',
     name='scatterd',
     version=new_version,
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     description="scatterd is an easy and fast way of creating scatter plots.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://erdogant.github.io/scatterd",
	 download_url = 'https://github.com/erdogant/scatterd/archive/'+new_version+'.tar.gz',
     packages=setuptools.find_packages(), # Searches throughout all dirs for files to include
     include_package_data=True, # Must be true to include files depicted in MANIFEST.in
     license_files=["LICENSE"],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
