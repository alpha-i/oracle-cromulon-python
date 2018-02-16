from setuptools import setup
from setuptools import find_packages


setup(
    name='alphai_cromulon_oracle',
    version='0.0.1',
    description='Alpha-I Cromulon',
    author='Sreekumar Thaithara Balan, Christopher Bonnett, Fergus Simpson',
    author_email='sreekumar.balan@alpha-i.co, christopher.bonnett@alpha-i.co, fergus.simpson@alpha-i.co',
    packages=find_packages(exclude=['doc', 'tests*']),
    install_requires=[
        'alphai-time-series>=0.0.3',
        'pandas-market-calendars>=0.6',
        'alphai_covariance==0.1.4',
        'alphai-data-sources>=1.0.1',
        'tensorflow>=1.4.0',
        'numpy>=1.12.0',
        'pandas==0.18.1',
        'scikit-learn',
        'alphai_feature_generation==1.4.0-dev-gym'
    ],
    dependency_links=[
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_covariance/',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_finance/',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_time_series/',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai-data-sources/',
        'https://pypi.fury.io/zNzsk7gQsYY335HLzW9x/alpha-i/alphai_feature_generation/'
    ]
)
