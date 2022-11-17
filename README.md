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

Train Hetu resnet on gpu:

```bash
bash examples/cnn/scripts/hetu_1gpu.sh resnet18 CIFAR10
```

Train Hetu resnet with allreduce on 8 gpus:

```bash
bash examples/cnn/scripts/hetu_8gpu.sh resnet18 CIFAR10
```

Train Hetu BERT base model on gpu:

```bash
cd examples/nlp/bert && bash scripts/create_datasets_from_start.sh # Dataset preparing
bash scripts/train_hetu_bert_base.sh
```

Train Hetu BERT base model with allreduce on 4 gpus:

```bash
cd examples/nlp/bert && bash scripts/create_datasets_from_start.sh # Dataset preparing
bash scripts/train_hetu_bert_base_dp.sh
```

Train Hetu Wide & Deep model on gpu:

```bash
bash examples/ctr/tests/local_wdl_adult.sh
```

Train Hetu Wide & Deep model with allreduce on 8 gpus:

```bash
bash examples/ctr/tests/dp_wdl_adult.sh
```


## More Examples
Please refer to examples directory, which contains CNN, NLP, CTR, GNN training scripts. For distributed training, please refer to CTR and GNN tasks.

## Community
* Email: xupeng.miao@pku.edu.cn
* Hetu homepage: https://hetu-doc.readthedocs.io
* [Committers & Contributors](COMMITTERS.md)
* [Contributing to Hetu](CONTRIBUTING.md)
* [Development plan](https://hetu-doc.readthedocs.io/en/latest/plan.html)


## Enterprise Users

If you are enterprise users and find Hetu is useful in your work, please let us know, and we are glad to add your company logo here.

<img src="./img/tencent.png" width = "200"/>
<img src="./img/alibabacloud.png" width = "200"/>
<img src="./img/kuaishou.png" width = "200"/>


## License

The entire codebase is under [license](LICENSE)

## Papers
  1. Xupeng Miao, Lingxiao Ma, Zhi Yang, Yingxia Shao, Bin Cui, Lele Yu, Jiawei Jiang. [CuWide: Towards Efficient Flow-based Training for Sparse Wide Models on GPUs](https://ieeexplore.ieee.org/document/9261124). TKDE 2021, ICDE 2021
  2. Xupeng Miao, Xiaonan Nie, Yingxia Shao, Zhi Yang, Jiawei Jiang, Lingxiao Ma, Bin Cui. [Heterogeneity-Aware Distributed Machine Learning Training via Partial Reduce](https://doi.org/10.1145/3448016.3452773). SIGMOD 2021
  3. Xupeng Miao, Hailin Zhang, Yining Shi, Xiaonan Nie, Zhi Yang, Yangyu Tao, Bin Cui. [HET: Scaling out Huge Embedding Model Training via Cache-enabled Distributed Framework](https://arxiv.org/abs/2112.07221). VLDB 2022, ChinaSys 2021 Winter.
  4. Xupeng Miao, Yining Shi, Hailin Zhang, Xin Zhang, Xiaonan Nie, Zhi Yang, Bin Cui. [HET-GMP: a Graph-based System Approach to Scaling Large Embedding Model Training](https://dl.acm.org/doi/10.1145/3514221.3517902). SIGMOD 2022.
  5. Xiaonan Nie, Xupeng Miao, Zhi Yang, Bin Cui. TSplit: Fine-grained GPU Memory Management for Efficient DNN Training via Tensor Splitting. ICDE 2022.
  6. Sicong Dong, Xupeng Miao, Pengkai Liu, Xin Wang, Bin Cui, Jianxin Li. HET-KG: Communication-Efficient Knowledge Graph Embedding Training via Hotness-Aware Cache. ICDE 2022.
  5. Xupeng Miao, Yujie Wang, Jia Shen, Yingxia Shao, Bin Cui. Graph Neural Network Training Acceleration over Multi-GPUs. Journal of Software (Chinese).
  6. Xiaonan Nie, Shijie Cao, Xupeng Miao, Lingxiao Ma, Jilong Xue, Youshan Miao, Zichao Yang, Zhi Yang, Bin Cui. [Dense-to-Sparse Gate for Mixture-of-Experts](https://arxiv.org/abs/2112.14397). arXiv 2021.
  7. Renrui Zhang, Ziyu Guo, Wei Zhang, Kunchang Li, Xupeng Miao, Bin Cui, Yu Qiao, Peng Gao, Hongsheng Li. [PointCLIP: Point Cloud Understanding by CLIP](https://arxiv.org/abs/2112.02413). CVPR 2022.
  8. coming soon

## Cite

If you use Hetu in a scientific publication, we would appreciate citations to the following paper:
```
 @article{miao2021het,
   title={HET: Scaling out Huge Embedding Model Training via Cache-enabled Distributed Framework},
   author={Miao, Xupeng and Zhang, Hailin and Shi, Yining and Nie, Xiaonan and Yang, Zhi and Tao, Yangyu and Cui, Bin},
   journal={arXiv preprint arXiv:2112.07221},
   year={2021}
 }
```

## Acknowledgements

We learned and borrowed insights from a few open source projects including [TinyFlow](https://github.com/tqchen/tinyflow), [autodist](https://github.com/petuum/autodist), [tf.distribute](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/distribute) and [Angel](https://github.com/Angel-ML/angel).

## Appendix
The prerequisites for different modules in Hetu is listed as follows:
  ```
  "*" means you should prepare by yourself, while others support auto-download
  
  Hetu: OpenMP(*), CMake(*)
  Hetu (version mkl): MKL 1.6.1
  Hetu (version gpu): CUDA 10.1(*), CUDNN 7.5(*), CUB 1.12.1(*)
  Hetu (version all): both

  Hetu-AllReduce: MPI 3.1, NCCL 2.8(*), this module needs GPU version

  Hetu-PS: Protobuf(*), ZeroMQ 4.3.2

  Hetu-Geometric: Pybind11(*), Metis(*)

  Hetu-Cache: Pybind11(*), this module needs PS module

  ##################################################################
  Tips for preparing the prerequisites
  
  Preparing CUDA, CUDNN, CUB, NCCL(NCCl is already in conda environment):
  1. download from https://developer.nvidia.com 
  2. download CUB from https://github.com/NVIDIA/cub/releases/tag/1.12.1
  3. install
  4. modify paths in cmake/config.cmake if necessary
  
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
