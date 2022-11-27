python -m torch.distributed.launch --nproc_per_node=8 --master_port 9996 test_allreduce.py \
--global_tp_deg 8 \
--global_tp_consec 1 \
--pp_deg 1

python -m torch.distributed.launch --nproc_per_node=8 --master_port 9996 test_allreduce.py \
--global_tp_deg 4 \
--global_tp_consec 0 \
--pp_deg 1

python -m torch.distributed.launch --nproc_per_node=8 --master_port 9996 test_allreduce.py \
--global_tp_deg 4 \
--global_tp_consec 1 \
--pp_deg 1

python -m torch.distributed.launch --nproc_per_node=8 --master_port 9996 test_allreduce.py \
--global_tp_deg 2 \
--global_tp_consec 0 \
--pp_deg 1

python -m torch.distributed.launch --nproc_per_node=8 --master_port 9996 test_allreduce.py \
--global_tp_deg 2 \
--global_tp_consec 1 \
--pp_deg 1

python -m torch.distributed.launch --nproc_per_node=4 --master_port 9996 test_allreduce.py \
--global_tp_deg 4 \
--global_tp_consec 1 \
--pp_deg 2

python -m torch.distributed.launch --nproc_per_node=4 --master_port 9996 test_allreduce.py \
--global_tp_deg 2 \
--global_tp_consec 0 \
--pp_deg 2

python -m torch.distributed.launch --nproc_per_node=4 --master_port 9996 test_allreduce.py \
--global_tp_deg 2 \
--global_tp_consec 1 \
--pp_deg 2

python -m torch.distributed.launch --nproc_per_node=2 --master_port 9996 test_allreduce.py \
--global_tp_deg 2 \
--global_tp_consec 1 \
--pp_deg 4