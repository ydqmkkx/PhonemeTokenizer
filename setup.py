from setuptools import setup, find_packages

setup(
    name='phoneme-tokenizer',
    version='1.0.0',
    packages=find_packages(),
    package_data={
        'PhonemeTokenizer.g2p': [
            'checkpoint20.npz', 
            'homographs.en',
            'averaged_perceptron_tagger.pickle',
            'cmudict'],
    },
    install_requires=[
        'inflect>=0.3.1',
        'nltk==3.7',
        'numpy>=1.13.1',
        'torch',
        'transformers',
    ],
    author='Dong-Yang',
    author_email='ydqmkkx@gmail.com',
    description='A package for phoneme tokenizer.',
    url='https://github.com/ydqmkkx/PhonemeTokenizer',
    keywords='phoneme tokenizer',
)