## Usage
This directory contains examples using `heturun` command.

* Data Parallel (MLP model and WDL model):
```bash
# Local Data Parallel Using AllReduce
heturun -c local_allreduce.yml python run_mlp.py --config lar

# Local Data Parallel Using AllReduce for Dense Parameters and PS for Sparse(Embedding) Parameters
heturun -c local_ps.yml python run_wdl.py --config lhy

# Local Data Parallel Using PS
heturun -c local_ps.yml python run_mlp.py --config lps
heturun -c local_ps.yml python run_wdl.py --config lps

# Distributed Data Parallel Using AllReduce
heturun -c remote_allreduce.yml python run_mlp.py --config rar

# Distributed Data Parallel Using AllReduce for Dense Parameters and PS for Sparse(Embedding) Parameters
heturun -c remote_ps.yml python run_wdl.py --config rhy

# Distributed Data Parallel Using PS
heturun -c remote_ps.yml python run_mlp.py --config rps
heturun -c remote_ps.yml python run_wdl.py --config rps
```

* For other parallel schemes, please refer to `parallel` directory.
