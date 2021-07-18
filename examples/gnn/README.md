# GNN Examples (with Distributed Settings)

## Structure
```
- gnn
    - gnn_tools/                  scripts to prepare data and other
    - config/                 distributed configurations
    - gnn_model/              gnn models
    - run_dist.py           train gnn models in ps setting
    - run_dist_hybrid.py    train gnn models in hybrid setting
    - run_single.py         train with a single gpu

```

## Configuration file explained

We use a simple yaml file to specify the run configuration.

```yaml
shared :
  DMLC_PS_ROOT_URI : 127.0.0.1
  DMLC_PS_ROOT_PORT : 13100
  DMLC_NUM_WORKER : 4
  DMLC_NUM_SERVER : 1
launch :
  worker : 4
  server : 1
  graph_server : 1
  scheduler : true
```

The 4 k-v pair in "shared" are used for PS-lite parameter server and will be added into environment. When running on a cluster, you should change "DMLC_PS_ROOT_URI" into an available IP address in the cluster.

The difference of GNN model and other models is that we need to launch a set of graph servers to carry out graph sampling. Note that the total number of graph server MUST be equal to the graph partition number. It is recommended that you partition the graph into the number of machines and launch one graph server on each machine.

Note that there should be only 1 scheduler and should only be launched on the machine with DMLC_PS_ROOT_URI.

Note that the launch automatically select network interface for you. If this fails, try adding "DMLC_INTERFACE : eth0" to select the right network device.

##  Prepare graph datasets

1. Prepare Normal dataset (use dense feature and no embedding)

   ```shell
python3 -m graphmix.partition [-d DatasetName] -n4 -p ~/yourDataPath
   ```

   We currently have the following dataset Cora, PubMed, Reddit, Flickr, Yelp, ogbn-products, ogbn-arxiv.

2. Prepare ogbn-mag or Reddit dataset (with sparse embedding)

   Then you can use the following command to partition the graph into 4 parts for 4-workers to use.

   ```bash
   python3 gnn_tools/part_graph.py [-d DatasetName] -n 4 -p ~/yourDataPath
   ```

   Also note that if you want to train on K node, replace the -n 4 with -n K.

3. Prepare Amazon dataset: This dataset is introduced in the cluster-GCN paper and there are two file to be downloaded: [metadata.json](https://drive.google.com/file/d/0B2jJQxNRDl_rVVZCdWVnYmUyRDg) and [map_files](https://drive.google.com/file/d/0B3lPMIHmG6vGd2U3VHB0Wkk4cGM). Once you download and extract the files and put them together under gnn_tools directory you can run

   ```bash
   python3 prepare_amazon_dataset.py
   ```

   Note that you need nltk installed in your environment to run this script and this will take a while.

   After running the script, you will get the two output file: graph.npz and sparsefeature.npy. Put them in the right place.

   ```bash
   mkdir -p ~/.graphmix_dataset/AmazonSparse
   mv graph.npz sparsefeature.npy ~/.graphmix_dataset/AmazonSparse
   ```

   Finally, use the part_graph.py to partition the graph

   ```
   python3 gnn_tools/part_graph.py -d AmazonSparse -n 4 -p ~/yourDataPath
   ```

## Training GNN Models

After you have prepare one graph dataset, you can start training Embedding Models on graph datasets. We take Reddit as an example.

To train on PS communication mode. Run

```
python3 run_dist.py [configfile] -p ~/yourDataPath/Reddit [--dense]
```

To train on Hybrid communication mode. Run

```
mpirun -np 4 --allow-run-as-root python3 run_dist_hybrid.py [configfile] -p ~/yourDataPath/Reddit [--dense]
```

When running on Hybrid mode, you will also have to launch some servers and scheduler seperately

```
python3 run_dist_hybrid.py [configfile] -p ~/yourDataPath/Reddit --server
```

A --dense argument is used if you are training with a normal dataset (with dense feature).

## Train with a single card

This time you will have to run partition as we mentioned before with n=1. After that, run

```shell
python3 run_single.py -p ~/yourDataPath/Reddit [--dense]
```

