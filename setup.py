from distutils.core import setup

setup(
	name='PyTrA',
	version='1.0',
	author='Jacob Martin',
	author_email='nzjakemartin@gmail.com',
	scripts=['PyTrA'],
	licence='LICENSE.txt',
	description='Python based ultrafast transient absorption spectroscopy data analysis',
	long_description=open('README.txt').read(),
	install_requires=[
		"pymc >= 1.1.1",
		"pymodelfit >= 0.2dev",
	]
	)