if [ ! -d results ];then
    mkdir results
else
    echo results directory exists
fi

heturun -c config3.yml python test_model_cnn_base.py --save --log results/base.npy

heturun -c config4.yml python test_model_cnn.py --split left   --log results/res0.npy
heturun -c config4.yml python test_model_cnn.py --split middle --log results/res1.npy
heturun -c config4.yml python test_model_cnn.py --split right  --log results/res2.npy

heturun -c config6.yml python test_model_cnn_complex.py --split 0 --log results/res3.npy
heturun -c config6.yml python test_model_cnn_complex.py --split 1 --log results/res4.npy
heturun -c config6.yml python test_model_cnn_complex.py --split 2 --log results/res5.npy
heturun -c config6.yml python test_model_cnn_complex.py --split 3 --log results/res6.npy
heturun -c config6.yml python test_model_cnn_complex.py --split 4 --log results/res7.npy

python validate_results.py 8 --rtol 1e-5

rm results/*

heturun -c config3.yml python test_model_cnn_base.py --save --trans --log results/base.npy

heturun -c config8.yml python test_model_cnn_complex.py --split 0 --split2 left   --log results/res0.npy
heturun -c config8.yml python test_model_cnn_complex.py --split 1 --split2 left   --log results/res1.npy
heturun -c config8.yml python test_model_cnn_complex.py --split 2 --split2 left   --log results/res2.npy
heturun -c config8.yml python test_model_cnn_complex.py --split 3 --split2 left   --log results/res3.npy
heturun -c config8.yml python test_model_cnn_complex.py --split 4 --split2 left   --log results/res4.npy
heturun -c config8.yml python test_model_cnn_complex.py --split 0 --split2 middle --log results/res5.npy
heturun -c config8.yml python test_model_cnn_complex.py --split 1 --split2 middle --log results/res6.npy
heturun -c config8.yml python test_model_cnn_complex.py --split 2 --split2 middle --log results/res7.npy
heturun -c config8.yml python test_model_cnn_complex.py --split 3 --split2 middle --log results/res8.npy
heturun -c config8.yml python test_model_cnn_complex.py --split 4 --split2 middle --log results/res9.npy
heturun -c config8.yml python test_model_cnn_complex.py --split 0 --split2 right  --log results/res10.npy
heturun -c config8.yml python test_model_cnn_complex.py --split 1 --split2 right  --log results/res11.npy
heturun -c config8.yml python test_model_cnn_complex.py --split 2 --split2 right  --log results/res12.npy
heturun -c config8.yml python test_model_cnn_complex.py --split 3 --split2 right  --log results/res13.npy
heturun -c config8.yml python test_model_cnn_complex.py --split 4 --split2 right  --log results/res14.npy

heturun -c config8.yml python test_model_cnn_complex.py --revert --split 0 --split2 left   --log results/res15.npy
heturun -c config8.yml python test_model_cnn_complex.py --revert --split 1 --split2 left   --log results/res16.npy
heturun -c config8.yml python test_model_cnn_complex.py --revert --split 2 --split2 left   --log results/res17.npy
heturun -c config8.yml python test_model_cnn_complex.py --revert --split 3 --split2 left   --log results/res18.npy
heturun -c config8.yml python test_model_cnn_complex.py --revert --split 4 --split2 left   --log results/res19.npy
heturun -c config8.yml python test_model_cnn_complex.py --revert --split 0 --split2 middle --log results/res20.npy
heturun -c config8.yml python test_model_cnn_complex.py --revert --split 1 --split2 middle --log results/res21.npy
heturun -c config8.yml python test_model_cnn_complex.py --revert --split 2 --split2 middle --log results/res22.npy
heturun -c config8.yml python test_model_cnn_complex.py --revert --split 3 --split2 middle --log results/res23.npy
heturun -c config8.yml python test_model_cnn_complex.py --revert --split 4 --split2 middle --log results/res24.npy
heturun -c config8.yml python test_model_cnn_complex.py --revert --split 0 --split2 right  --log results/res25.npy
heturun -c config8.yml python test_model_cnn_complex.py --revert --split 1 --split2 right  --log results/res26.npy
heturun -c config8.yml python test_model_cnn_complex.py --revert --split 2 --split2 right  --log results/res27.npy
heturun -c config8.yml python test_model_cnn_complex.py --revert --split 3 --split2 right  --log results/res28.npy
heturun -c config8.yml python test_model_cnn_complex.py --revert --split 4 --split2 right  --log results/res29.npy

heturun -c config6.yml python test_model_cnn.py --split left   --split2 left   --log results/res30.npy
heturun -c config6.yml python test_model_cnn.py --split left   --split2 middle --log results/res31.npy
heturun -c config6.yml python test_model_cnn.py --split left   --split2 right  --log results/res32.npy
heturun -c config6.yml python test_model_cnn.py --split middle --split2 left   --log results/res33.npy
heturun -c config6.yml python test_model_cnn.py --split middle --split2 middle --log results/res34.npy
heturun -c config6.yml python test_model_cnn.py --split middle --split2 right  --log results/res35.npy
heturun -c config6.yml python test_model_cnn.py --split right  --split2 left   --log results/res36.npy
heturun -c config6.yml python test_model_cnn.py --split right  --split2 middle --log results/res37.npy
heturun -c config6.yml python test_model_cnn.py --split right  --split2 right  --log results/res38.npy

python validate_results.py 39 --rtol 1e-5

rm results/*
rm std/*
