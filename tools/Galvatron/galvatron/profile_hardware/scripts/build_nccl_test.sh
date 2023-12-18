if [ "$USE_EXPORT_VARIABLE" = "1" ]; then
echo "USE_EXPORT_VARIABLE is set to 1, using the exported variables."
else
echo "USE_EXPORT_VARIABLE is not set to 1, using the variables defined in script."
MPI_PATH=/usr/local/mpi/
MAKE_MPI=1
fi

cd ../site_package/nccl-tests
if [ "$MAKE_MPI" = "1" ]; then
echo 'Building nccl-tests with MPI.'
make MPI=1 MPI_HOME=${MPI_PATH}
else
echo 'Building nccl-tests without MPI.'
make
fi