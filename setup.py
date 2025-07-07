from setuptools import find_packages,setup

setup(
    name='MCQGenrator',
    version='0.0.1',
    author='Nutnell',
    author_email='nutnell00@gmail.com',
    install_requires=["openai","langchain","streamlit","python-dotenv","Pypdf"],
    packages=find_packages()
)