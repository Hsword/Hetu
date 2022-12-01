cd ./overlap_test
sh test_overlap.sh
cd ..
cd ./bandwidth_test_dist
sh test_allreduce_2nodes_all.sh
sh test_p2p_2nodes_all.sh
cd ..