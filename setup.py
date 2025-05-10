from setuptools import setup, find_packages

setup(
    name="pytc",
    version="0.1.0",
    description="Python TransCorrelation package",
    author="Ke Liao",
    author_email="ke.liao.whu@gmail.com",
    url="https://github.com/nickirk/pytc",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "jax",
        "optax",
        "kfac_jax",
        "flax",
        "tqdm",
        "pyscf",
    ],
    extras_require={
        "dev": [
            "unittest"
        ],
        "gpu": [
            "jaxlib[cuda]",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)