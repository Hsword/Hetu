if [ "$USE_EXPORT_VARIABLE" = "1" ]; then
echo "USE_EXPORT_VARIABLE is set to 1, using the exported variables."
else
echo "USE_EXPORT_VARIABLE is not set to 1, using the variables defined in script."
NUM_GPUS_PER_NODE=8
OVERLAP_TIME_MULTIPLY=4
fi

ARGS="-m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS_PER_NODE} \
--master_port 9999 \
profile_overlap.py \
--overlap_time_multiply ${OVERLAP_TIME_MULTIPLY}"

echo "Running: python3 ${ARGS}"
python3 ${ARGS}