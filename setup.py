from setuptools import setup, find_packages

with open("requirements.txt", "r") as file:
    lines = file.readlines()
reqs =[req for req in lines if "#" not in req]

setup(name="sspo", packages=find_packages(), install_requires=reqs)
