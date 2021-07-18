## Usage
* Complex Pipeline Parallel (not using heturun):
```bash
mpirun --allow-run-as-root --tag-output -np 8 python complex_pipeline_mlp.py
```

* Simple Pipeline Parallel:
```bash
heturun -c config8.yml python simple_pipeline_mlp.py
```

* Data + Pipeline Parallel:
```bash
heturun -c config8.yml python data_pipeline_mlp.py
```

* Multiple Machine Data + Pipeline Parallel:
```bash
heturun -c dist_config8.yml python dist_data_pipeline_mlp.py
```

* Test Model Parallel (the following commands should give the same results):
```bash
heturun -c config3.yml python test_model_mlp_base.py --save
heturun -c config4.yml python test_model_mlp.py --split left
heturun -c config4.yml python test_model_mlp.py --split right
heturun -c config4.yml python test_model_mlp.py --split middle
```

* Data + Model (+ Pipeline) Parallel:
```bash
heturun -c config8.yml python data_model_pipeline_mlp.py --split left
heturun -c config8.yml python data_model_pipeline_mlp.py --split right
heturun -c config8.yml python data_model_pipeline_mlp.py --split middle
```
