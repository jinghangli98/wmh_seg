from setuptools import setup, find_packages

setup(
    name='wmh_seg',
    version='1.6.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={'wmh_seg': ['nnunet_assets/*/*.json']},
    description='White matter hyperintensity segmentation for in-vivo and ex-vivo MRI',
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
        "timm==0.6.12",
        'numpy',
        'segmentation-models-pytorch==0.3.2',
        'huggingface_hub',
        'nnunetv2',
    ],
    entry_points={
        'console_scripts': [
            'wmh_seg=wmh_seg.cli:main',
        ],
    },
)
