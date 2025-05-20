from setuptools import setup, find_packages

setup(
    name='llama_cpp',
    version='0.1.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Add any dependencies here
    ],
    entry_points={
        'console_scripts': [
            # Add any command line scripts here
        ],
    },
    author='Shamit Verma',
    author_email='oss@shamit.in',
    description='A Python package for Llama CPP.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/shamitv/llama_cpp',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
