import setuptools



setuptools.setup(
    name="historydag",
    version=0.1,
    author="Will Dumm",
    author_email="wdumm88@gmail.com",
    description="Basic history DAG implementation",
    packages=['historydag'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "ete3",
        "biopython",
    ],
)
