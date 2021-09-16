<div align=center>
<img src="./img/hetu.png" width="300" />
</div>

# HETU

<!--- [![license](https://img.shields.io/github/license/apache/zookeeper?color=282661)](LICENSE) --->

[Documentation](https://hetu-doc.readthedocs.io) | [Examples](https://hetu-doc.readthedocs.io/en/latest/Overview/performance.html)

Hetu is a high-performance distributed deep learning system targeting trillions of parameters DL model training, developed by <a href="http://net.pku.edu.cn/~cuibin/" target="_blank" rel="nofollow">DAIR Lab</a> at Peking University. It takes account of both high availability in industry and innovation in academia, which has a number of advanced characteristics:

- Applicability. DL model definition with standard dataflow graph; many basic CPU and GPU operators; efficient implementation of more than plenty of DL models and at least popular 10 ML algorithms.

- Efficiency. Achieve at least 30% speedup compared to TensorFlow on DNN, CNN, RNN benchmarks.

- Flexibility. Supporting various parallel training protocols and distributed communication architectures, such as Data/Model/Pipeline parallel; Parameter server & AllReduce.

- Scalability. Deployment on more than 100 computation nodes; Training giant models with trillions of model parameters, e.g., Criteo Kaggle, Open Graph Benchmark

- Agility. Automatically ML pipeline: feature engineering, model selection, hyperparameter search.

We welcome everyone interested in machine learning or graph computing to contribute codes, create issues or pull requests. Please refer to [Contribution Guide](CONTRIBUTING.md) for more details.

## Installation
1. Clone the repository.

2. Prepare the environment. We use Anaconda to manage packages. The following command create the conda environment to be used:`conda env create -f environment.yml`. Please prepare Cuda toolkit and CuDNN in advance.

3. We use CMake to compile Hetu. Please copy the example configuration for compilation by `cp cmake/config.example.cmake cmake/config.cmake`. Users can modify the configuration file to enable/disable the compilation of each module. For advanced users (who not using the provided conda environment), the prerequisites for different modules in Hetu is listed in appendix.

```bash
# modify paths and configurations in cmake/config.cmake

# generate Makefile
mkdir build && cd build && cmake ..

# compile
# make all
make -j 8
# make hetu, version is specified in cmake/config.cmake
make hetu -j 8
# make allreduce module
make allreduce -j 8
# make ps module
make ps -j 8
# make geometric module
make geometric -j 8
# make hetu-cache module
make hetu_cache -j 8
```

4. Prepare environment for running. Edit the hetu.exp file and set the environment path for python and the path for executable mpirun if necessary (for advanced users not using the provided conda environment). Then execute the command `source hetu.exp` .



## Usage

Train logistic regression on gpu:

```bash
bash examples/cnn/scripts/hetu_1gpu.sh logreg MNIST
```

Train a 3-layer mlp on gpu:

```bash
bash examples/cnn/scripts/hetu_1gpu.sh mlp CIFAR10
```

Train a 3-layer cnn with gpu:

```bash
bash examples/cnn/scripts/hetu_1gpu.sh cnn_3_layers MNIST
```

Train a 3-layer mlp with allreduce on 8 gpus (use mpirun):

```bash
bash examples/cnn/scripts/hetu_8gpu.sh mlp CIFAR10
```

Train a 3-layer mlp with PS on 1 server and 2 workers:

```bash
# in the script we launch the scheduler and server, and two workers
bash examples/cnn/scripts/hetu_2gpu_ps.sh mlp CIFAR10
```


## More Examples
Please refer to examples directory, which contains CNN, NLP, CTR, GNN training scripts. For distributed training, please refer to CTR and GNN tasks.

## Community
* Email: xupeng.miao@pku.edu.cn
* Hetu homepage: https://hetu-doc.readthedocs.io
* [Committers & Contributors](COMMITTERS.md)
* [Contributing to Hetu](CONTRIBUTING.md)
* [Development plan](https://hetu-doc.readthedocs.io/en/latest/plan.html)

<!---
## Enterprise Users

If you are enterprise users and find Hetu is useful in your work, please let us know, and we are glad to add your company logo here.

<img src="./img/tencent.png" width = "200"/>
<img src="./img/alibabacloud.png" width = "200"/>
<img src="./img/kuaishou.png" width = "200"/>
--->

## License

The entire codebase is under [license](LICENSE)

## Papers
  1. Xupeng Miao, Linxiao Ma, Zhi Yang, Yingxia Shao, Bin Cui, Lele Yu, Jiawei Jiang. [CuWide: Towards Efficient Flow-based Training for Sparse Wide Models on GPUs](https://ieeexplore.ieee.org/document/9261124). TKDE 2021, ICDE 2021
  2. Xupeng Miao, Xiaonan Nie, Yingxia Shao, Zhi Yang, Jiawei Jiang, Lingxiao Ma, Bin Cui. [Heterogeneity-Aware Distributed Machine Learning Training via Partial Reduce](https://doi.org/10.1145/3448016.3452773). SIGMOD 2021
  3. Xupeng Miao, Hailin Zhang, Yining Shi, Xiaonan Nie, Zhi Yang, Yangyu Tao, Bin Cui. HET: Scaling out Huge Embedding Model Training via Cache-enabled Distributed Framework. VLDB 2021
  4. coming soon

## Acknowledgements

We learned and borrowed insights from a few open source projects including [TinyFlow](https://github.com/tqchen/tinyflow), [autodist](https://github.com/petuum/autodist), [tf.distribute](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/distribute) and [Angel](https://github.com/Angel-ML/angel).

## Appendix
The prerequisites for different modules in Hetu is listed as follows:
  ```
  "*" means you should prepare by yourself, while others support auto-download
  
  Hetu: OpenMP(*), CMake(*)
  Hetu (version mkl): MKL 1.6.1
  Hetu (version gpu): CUDA 10.1(*), CUDNN 7.5(*)
  Hetu (version all): both

  Hetu-AllReduce: MPI 3.1, NCCL 2.8(*), this module needs GPU version

  Hetu-PS: Protobuf(*), ZeroMQ 4.3.2

  Hetu-Geometric: Pybind11(*), Metis(*)

  Hetu-Cache: Pybind11(*), this module needs PS module

  ##################################################################
  Tips for preparing the prerequisites
  
  Preparing CUDA, CUDNN, NCCL(NCCl is already in conda environment):
  1. download from https://developer.nvidia.com
  2. install
  3. modify paths in cmake/config.cmake if necessary
  
  Preparing OpenMP:
  Your just need to ensure your compiler support openmp.

  Preparing CMake, Protobuf, Pybind11, Metis:
  Install by anaconda: 
  conda install cmake=3.18 libprotobuf pybind11=2.6.0 metis

  Preparing OpenMPI (not necessary):
  install by anaconda: `conda install -c conda-forge openmpi=4.0.3`
  or
  1. download from https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz
  2. build openmpi by `./configure /path/to/build && make -j8 && make install`
  3. modify MPI_HOME to /path/to/build in cmake/config.cmake

  Preparing MKL (not necessary):
  install by anaconda: `conda install -c conda-forge onednn`
  or
  1. download from https://github.com/intel/mkl-dnn/archive/v1.6.1.tar.gz
  2. build mkl by `mkdir /path/to/build && cd /path/to/build && cmake /path/to/root && make -j8` 
  3. modify MKL_ROOT to /path/to/root and MKL_BUILD to /path/to/build in cmake/config.cmake 

  Preparing ZeroMQ (not necessary):
  install by anaconda: `conda install -c anaconda zeromq=4.3.2`
  or
  1. download from https://github.com/zeromq/libzmq/releases/download/v4.3.2/zeromq-4.3.2.zip
  2. build zeromq by 'mkdir /path/to/build && cd /path/to/build && cmake /path/to/root && make -j8`
  3. modify ZMQ_ROOT to /path/to/build in cmake/config.cmake
  ```
