## Structure
```
- moe
    - scripts/              Test scripts
    - test_moe_top.py       TopK MoE
    - test_moe_hash.py      Hash Layer MoE
    - test_moe_ktop1.py     KTop1 MoE
    - test_moe_base.py      BASE Layer MoE
    - test_moe_sam.py       Switch and Mixture MoE
    - 
```
## Usage
Here are some examples of running scripts.
```bash
bash scripts/run_top1.sh 
```
Change ht.alltoall\_op into ht.halltoall\_op in the model definition(located in Hetu/python/hetu/layers) to use Hierarchical AllToAll. 
