from setuptools import setup, find_packages

setup(
    name="arc_solver",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "trace_visualizer=tools.trace_visualizer:main",
        ]
    },
)
