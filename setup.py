from setuptools import find_packages, setup  # type: ignore [import-untyped]

setup(
    name="volatility_arbitrage",
    version="0.1.0",
    packages=find_packages(include=["volatility_arbitrage", "volatility_arbitrage.*"]),
)
