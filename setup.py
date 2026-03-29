from setuptools import setup, find_packages

setup(
    name="eye-tracking-benchmark",
    version="1.0.0",
    description="Standardized benchmarking tool for eye-tracking accuracy and precision validation",
    author="Your Name",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "streamlit>=1.28.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "tqdm>=4.65.0",
        "Pillow>=10.0.0",
        "jinja2>=3.1.0",
    ],
    entry_points={
        "console_scripts": [
            "etbench=scripts.run_benchmark:main",
        ],
    },
)
