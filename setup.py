import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
        name='sgpvae',
        version='0.1',
        author='Matthew Ashman',
        author_email='mca39@cam.ac.uk',
        description='Sparse Gaussian process variational autoencoders',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/MattAshman/sgpvae',
        packages=setuptools.find_packages(),
        classifiers=[
                'Programming Language :: Python :: 3',
                'License :: OSI Approved :: MIT License',
                'Operating System :: OS Independent',
        ],
        python_requires='>3.6',
)

