"""
Customer Intelligence Platform Setup Script.

This file enables installing the backend packages and modules in editable mode.
"""

from setuptools import find_packages, setup

setup(
    name="customer-intelligence-platform",
    version="1.0.0",
    description="AI-Powered Customer Churn Prediction & Lifetime Value (LTV) Engine",
    author="CIP Engineering Team",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.28.0",
        "pydantic>=2.6.0",
        "pydantic-settings>=2.2.0",
        "SQLAlchemy>=2.0.28",
        "psycopg2-binary>=2.9.9",
        "pandas>=2.2.1",
        "numpy>=1.26.4",
        "scikit-learn>=1.4.1.post1",
        "python-dotenv>=1.0.1",
    ],
)
