from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
import pathlib
import os

try:
    import fused_dense_lib, dropout_layer_norm, rotary_emb, xentropy_cuda_lib
except ImportError:
    fused_dense_lib, dropout_layer_norm, rotary_emb, xentropy_cuda_lib = None, None, None, None
    

FLASH_ATTN_INSTALL = os.getenv("GALVATRON_FLASH_ATTN_INSTALL", "FALSE") == "TRUE"

here = pathlib.Path(__file__).parent.resolve()

class CustomInstall(install):
    def run(self):
        install.run(self)

        # custom install flash-attention cuda ops by running shell scripts
        if FLASH_ATTN_INSTALL:
            cwd = pathlib.Path.cwd()
            
            if fused_dense_lib is None or dropout_layer_norm is None or rotary_emb is None or xentropy_cuda_lib is None:
                self.spawn(["bash", cwd / "galvatron" / "scripts" / "flash_attn_ops_install.sh"])


class CustomBuildExt(build_ext):
    def run(self):
        import pybind11

        self.include_dirs.append(pybind11.get_include())

        build_ext.run(self)


# Define the extension module
dp_core_ext = Extension(
    'galvatron_dp_core',
    sources=['csrc/dp_core.cpp'],
    extra_compile_args=['-O3', '-Wall', '-shared', '-std=c++11', '-fPIC'],
    language='c++'
)

_deps = [
    "torch==2.0.1",
    "torchvision==0.15.2",
    "transformers>=4.31.0",
    "h5py>=3.6.0",
    "attrs>=21.4.0",
    "yacs>=0.1.8",
    "six>=1.15.0",
    "sentencepiece>=0.1.95"
]

if FLASH_ATTN_INSTALL:
    _deps.append("packaging")
    _deps.append("flash-attn>=2.0.8")

setup(
    name="hetu-galvatron",
    version="1.0.0",
    description="Galvatron, a Efficient Transformer Training Framework for Multiple GPUs Using Automatic Parallelism",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Yujie Wang, Shenhan Zhu",
    author_email="alfredwang@pku.edu.cn, shenhan.zhu@pku.edu.cn",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "figs",
            "*egg-info"
        )
    ),
    package_data={"": ["*.json"]},
    include_package_data=True,
    scripts=["galvatron/scripts/flash_attn_ops_install.sh"],
    python_requires=">=3.8",
    cmdclass={
        "install": CustomInstall,
        "build_ext": CustomBuildExt
    },
    install_requires=_deps,
    setup_requires=["pybind11>=2.9.1"],
    ext_modules=[dp_core_ext]
)
