from setuptools import setup, find_packages

setup(
    name="parmap",
    version="0.dev",
    packages=find_packages(),
    author=["Brian Wilson", "Derek Posselt", "Vishal Lall", "Diego Martinez"],
    author_email=[
        "bdwilson@jpl.nasa.gov",
        "derek.posselt@jpl.nasa.gov",
        "vishal.lall@jpl.nasa.gov",
        "diego.s.martinez@jpl.nasa.gov",
    ],
    description="Generalized parallel computing in one-line of Python.",
)