if [ ! -d results ];then
    mkdir results
else
    echo results directory exists
fi

heturun -c config3.yml python test_model_mlp_base.py --save --log results/base.npy
heturun -c config4.yml python test_model_mlp_param.py --split left   --log results/res0.npy
heturun -c config4.yml python test_model_mlp_param.py --split middle --log results/res1.npy
heturun -c config4.yml python test_model_mlp_param.py --split right  --log results/res2.npy
python validate_results.py 3

rm results/*

heturun -c config3.yml python test_model_mlp_base.py --save --trans --log results/base.npy
heturun -c config6.yml python test_model_mlp_param.py --split left   --split2 left   --log results/res0.npy
heturun -c config6.yml python test_model_mlp_param.py --split left   --split2 middle --log results/res1.npy
heturun -c config6.yml python test_model_mlp_param.py --split left   --split2 right  --log results/res2.npy
heturun -c config6.yml python test_model_mlp_param.py --split middle --split2 left   --log results/res3.npy
heturun -c config6.yml python test_model_mlp_param.py --split middle --split2 middle --log results/res4.npy
heturun -c config6.yml python test_model_mlp_param.py --split middle --split2 right  --log results/res5.npy
heturun -c config6.yml python test_model_mlp_param.py --split right  --split2 left   --log results/res6.npy
heturun -c config6.yml python test_model_mlp_param.py --split right  --split2 middle --log results/res7.npy
heturun -c config6.yml python test_model_mlp_param.py --split right  --split2 right  --log results/res8.npy
python validate_results.py 9

rm results/*
rm std/*
