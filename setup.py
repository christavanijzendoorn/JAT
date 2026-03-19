import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="JAT", # Replace with your own username
    version="0.0.1",
    author="Christa van IJzendoorn",
    author_email="c.o.vanijzendoorn@tudelft.nl",
    description="Jarkus Analysis Toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/christavanijzendoorn/JAT.git",
    packages=setuptools.find_packages(),
	install_requires=[
		'numpy',
		'pandas',
		'netCDF4',
		'scipy',
		'matplotlib',
		'pyyaml',],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
) 