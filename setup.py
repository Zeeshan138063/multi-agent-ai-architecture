"""
Setup script for Modular Agentic AI Architecture
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("modular_agentic_ai/requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="modular-agentic-ai",
    version="1.0.0",
    author="Zeehsan (PS.Brij Kishore Pandey)",
    author_email="",
    description="A modular, plugin-based architecture for building scalable agentic AI systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/modular-agentic-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.19.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "openai": [
            "openai>=1.0.0",
        ],
        "anthropic": [
            "anthropic>=0.8.0",
        ],
        "transformers": [
            "transformers>=4.25.0",
            "torch>=1.13.0",
            "tokenizers>=0.13.0",
        ],
        "vector": [
            "faiss-cpu>=1.7.0",
            "sentence-transformers>=2.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "modular-ai-demo=modular_agentic_ai.examples.modular_demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "modular_agentic_ai": [
            "config/*.yaml",
            "services/*/*.yaml",
        ],
    },
)