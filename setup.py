from setuptools import setup, find_packages

setup(
    name="vigil",
    version="1.0.0",
    description="Video Anomalous Activity Detection System",
    author="VIGIL Team",
    packages=find_packages(exclude=["tests*", "notebooks*", "training*"]),
    python_requires=">=3.9",
    install_requires=[
        "kagglehub>=0.2.0",
        "ultralytics>=8.0.0",
        "opencv-python-headless>=4.8.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "reportlab>=4.0.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "python-multipart>=0.0.6",
        "aiofiles>=23.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "anthropic>=0.20.0",
    ],
    entry_points={
        "console_scripts": [
            "vigil=run:main",
        ]
    },
)
