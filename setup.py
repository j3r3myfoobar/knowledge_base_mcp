from setuptools import setup, find_packages

setup(
    name="mcp_server",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "rank-bm25>=0.2.2",
    ],
)
