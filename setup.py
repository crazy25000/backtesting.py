import sys

if sys.version_info < (3, 8):
    sys.exit('ERROR: backtesting.py requires Python 3.8+')


if __name__ == '__main__':
    from setuptools import setup, find_packages

    setup(
        name='backtesting',
        description='Backtest trading strategies in Python',
        license='AGPL-3.0',
        packages=find_packages(),
        include_package_data=True,
        setup_requires=[
            'setuptools_git',
            'setuptools_scm',
        ],
        install_requires=[
            'numpy',
            'pandas >= 0.25.0, != 0.25.0',
            'bokeh >= 1.4.0',
            'tqdm >= 4.59.0',
        ],
        extras_require={
            'doc': [
                'ipykernel',  # for nbconvert
                'jupyter_client',  # for nbconvert
                'jupytext >= 1.3',
                'nbconvert',
                'pdoc3',
            ],
            'test': [
                'matplotlib',
                'scikit-learn',
                'scikit-optimize',
                'seaborn',
                'pytest',
            ],
            'dev': [
                'black',
                'coverage',
                'flake8',
                'isort',
                'mypy',
                'pre-commit',
                'tox',
            ],
        },
        test_suite='backtesting.test',
        python_requires='>=3.8',
    )
