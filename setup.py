import setuptools

with open('README.md','r') as file:
	long_description = file.read()


setuptools.setup(
	name = 'preprocess_kamalkumar_nitj',
	version = '0.0.8',
	author = 'kamal kumar',
	author_email = 'erkamalkumar.0001@gmail.com',
	description = 'This is preprocessing package',
	Long_description = long_description,
	Long_description_content_type = 'text/markdown',
	packages = setuptools.find_packages(),
	classifiers = [
	'Programming Language :: Python :: 3',
	'License :: OSI Aproved :: MIT License',
	"Operating System :: OS Independent"],
	python_requires = '>=3.5'
	)	