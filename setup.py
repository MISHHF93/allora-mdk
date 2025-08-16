from setuptools import setup, find_packages

setup(
    name="allora-mdk",
    version="0.1.0",
    description="Machine learning pipeline for forecasting using financial data",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/allora-mdk",  # Update if needed
    packages=find_packages(exclude=["tests", "notebooks"]),
    include_package_data=True,
    install_requires=[
        "pystan==2.19.1.1",
        "prophet",
        "scikit-learn",
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "statsmodels",
        "fastapi",
        "uvicorn",
        "torch",
        "xgboost",
        "yfinance",
        "requests",
        "python-dotenv",
        "PyYAML",
        "holidays",
        "joblib",
        "tqdm",
        "cmdstanpy",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "pylint",
            "pre-commit",
            "pipdeptree"
        ]
    },
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
)
