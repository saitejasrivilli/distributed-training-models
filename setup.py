from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="distributed-llm-training",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Distributed LLM Pre-training from Scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/distributed-llm-training",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
        ],
        "deepspeed": ["deepspeed>=0.12.0"],
        "wandb": ["wandb>=0.15.0"],
    },
)
