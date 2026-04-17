from setuptools import setup, find_packages

setup(
    name='wmh_seg',
    version='0.1.0',
    packages=find_packages(), 
    include_package_data=True,
    description='WMH segmentation for FLAIR images',
    author='Jinghang Li',
    author_email='jinghang.li@pitt.edu',
    url='https://github.com/jinghangli98/wmh_seg',  # Replace with your GitHub repo
    install_requires=[
        'tqdm',
        'torch',
        'torchvision',
        'nibabel',
        'einops',
        'torchio',
        'numpy==1.26.4',
        'segmentation-models-pytorch==0.3.2',
        'requests',
    ],
)
