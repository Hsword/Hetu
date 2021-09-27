SIMPLE_TEST=true
COMPLEX_MP_TEST=true
COMPLEX_MP_PP_TEST=true
REMOVE_RESULTS=true


if [ ! -d results ];then
    mkdir results
else
    echo results directory exists
fi


if $SIMPLE_TEST; then
    heturun -c config1.yml python test_mlp_base.py --save --log results/base.npy

    heturun -c config3.yml python test_mlp_pp.py --log results/res0.npy

    heturun -c config2.yml python test_mlp_mp.py --split left   --log results/res1.npy
    heturun -c config2.yml python test_mlp_mp.py --split middle --log results/res2.npy
    heturun -c config2.yml python test_mlp_mp.py --split right  --log results/res3.npy
    heturun -c config4.yml python test_mlp_mp.py --split 0      --log results/res4.npy
    heturun -c config4.yml python test_mlp_mp.py --split 1      --log results/res5.npy
    heturun -c config4.yml python test_mlp_mp.py --split 2      --log results/res6.npy
    heturun -c config4.yml python test_mlp_mp.py --split 3      --log results/res7.npy
    heturun -c config4.yml python test_mlp_mp.py --split 4      --log results/res8.npy

    heturun -c config4.yml python test_mlp_mp_pp.py --split left   --log results/res9.npy
    heturun -c config4.yml python test_mlp_mp_pp.py --split middle --log results/res10.npy
    heturun -c config4.yml python test_mlp_mp_pp.py --split right  --log results/res11.npy
    heturun -c config4.yml python test_mlp_mp_pp.py --split 0      --log results/res12.npy
    heturun -c config4.yml python test_mlp_mp_pp.py --split 1      --log results/res13.npy
    heturun -c config4.yml python test_mlp_mp_pp.py --split 2      --log results/res14.npy
    heturun -c config4.yml python test_mlp_mp_pp.py --split 3      --log results/res15.npy
    heturun -c config4.yml python test_mlp_mp_pp.py --split 4      --log results/res16.npy

    heturun -c config4.yml python test_mlp_mp.py --split 5      --log results/res17.npy
    heturun -c config4.yml python test_mlp_mp_pp.py --split 5      --log results/res18.npy

    python validate_results.py 19
fi


if $COMPLEX_MP_TEST; then
    heturun -c config1.yml python test_mlp_base.py --more --save --log results/base.npy

    heturun -c config3.yml python test_mlp_pp.py --more --log results/res0.npy

    heturun -c config2.yml python test_mlp_mp.py --split left   --split2 left --log results/res1.npy
    heturun -c config2.yml python test_mlp_mp.py --split middle --split2 left --log results/res2.npy
    heturun -c config2.yml python test_mlp_mp.py --split right  --split2 left --log results/res3.npy
    heturun -c config4.yml python test_mlp_mp.py --split 0      --split2 left --log results/res4.npy
    heturun -c config4.yml python test_mlp_mp.py --split 1      --split2 left --log results/res5.npy
    heturun -c config4.yml python test_mlp_mp.py --split 2      --split2 left --log results/res6.npy
    heturun -c config4.yml python test_mlp_mp.py --split 3      --split2 left --log results/res7.npy
    heturun -c config4.yml python test_mlp_mp.py --split 4      --split2 left --log results/res8.npy
    heturun -c config2.yml python test_mlp_mp.py --split left   --split2 right --log results/res9.npy
    heturun -c config2.yml python test_mlp_mp.py --split middle --split2 right --log results/res10.npy
    heturun -c config2.yml python test_mlp_mp.py --split right  --split2 right --log results/res11.npy
    heturun -c config4.yml python test_mlp_mp.py --split 0      --split2 right --log results/res12.npy
    heturun -c config4.yml python test_mlp_mp.py --split 1      --split2 right --log results/res13.npy
    heturun -c config4.yml python test_mlp_mp.py --split 2      --split2 right --log results/res14.npy
    heturun -c config4.yml python test_mlp_mp.py --split 3      --split2 right --log results/res15.npy
    heturun -c config4.yml python test_mlp_mp.py --split 4      --split2 right --log results/res16.npy
    heturun -c config2.yml python test_mlp_mp.py --split left   --split2 middle --log results/res17.npy
    heturun -c config2.yml python test_mlp_mp.py --split middle --split2 middle --log results/res18.npy
    heturun -c config2.yml python test_mlp_mp.py --split right  --split2 middle --log results/res19.npy
    heturun -c config4.yml python test_mlp_mp.py --split 0      --split2 middle --log results/res20.npy
    heturun -c config4.yml python test_mlp_mp.py --split 1      --split2 middle --log results/res21.npy
    heturun -c config4.yml python test_mlp_mp.py --split 2      --split2 middle --log results/res22.npy
    heturun -c config4.yml python test_mlp_mp.py --split 3      --split2 middle --log results/res23.npy
    heturun -c config4.yml python test_mlp_mp.py --split 4      --split2 middle --log results/res24.npy
    heturun -c config4.yml python test_mlp_mp.py --split left   --split2 0 --log results/res25.npy
    heturun -c config4.yml python test_mlp_mp.py --split middle --split2 0 --log results/res26.npy
    heturun -c config4.yml python test_mlp_mp.py --split right  --split2 0 --log results/res27.npy
    heturun -c config4.yml python test_mlp_mp.py --split 0      --split2 0 --log results/res28.npy
    heturun -c config4.yml python test_mlp_mp.py --split 1      --split2 0 --log results/res29.npy
    heturun -c config4.yml python test_mlp_mp.py --split 2      --split2 0 --log results/res30.npy
    heturun -c config4.yml python test_mlp_mp.py --split 3      --split2 0 --log results/res31.npy
    heturun -c config4.yml python test_mlp_mp.py --split 4      --split2 0 --log results/res32.npy
    heturun -c config4.yml python test_mlp_mp.py --split left   --split2 1 --log results/res33.npy
    heturun -c config4.yml python test_mlp_mp.py --split middle --split2 1 --log results/res34.npy
    heturun -c config4.yml python test_mlp_mp.py --split right  --split2 1 --log results/res35.npy
    heturun -c config4.yml python test_mlp_mp.py --split 0      --split2 1 --log results/res36.npy
    heturun -c config4.yml python test_mlp_mp.py --split 1      --split2 1 --log results/res37.npy
    heturun -c config4.yml python test_mlp_mp.py --split 2      --split2 1 --log results/res38.npy
    heturun -c config4.yml python test_mlp_mp.py --split 3      --split2 1 --log results/res39.npy
    heturun -c config4.yml python test_mlp_mp.py --split 4      --split2 1 --log results/res40.npy
    heturun -c config4.yml python test_mlp_mp.py --split left   --split2 2 --log results/res41.npy
    heturun -c config4.yml python test_mlp_mp.py --split middle --split2 2 --log results/res42.npy
    heturun -c config4.yml python test_mlp_mp.py --split right  --split2 2 --log results/res43.npy
    heturun -c config4.yml python test_mlp_mp.py --split 0      --split2 2 --log results/res44.npy
    heturun -c config4.yml python test_mlp_mp.py --split 1      --split2 2 --log results/res45.npy
    heturun -c config4.yml python test_mlp_mp.py --split 2      --split2 2 --log results/res46.npy
    heturun -c config4.yml python test_mlp_mp.py --split 3      --split2 2 --log results/res47.npy
    heturun -c config4.yml python test_mlp_mp.py --split 4      --split2 2 --log results/res48.npy
    heturun -c config4.yml python test_mlp_mp.py --split left   --split2 3 --log results/res49.npy
    heturun -c config4.yml python test_mlp_mp.py --split middle --split2 3 --log results/res50.npy
    heturun -c config4.yml python test_mlp_mp.py --split right  --split2 3 --log results/res51.npy
    heturun -c config4.yml python test_mlp_mp.py --split 0      --split2 3 --log results/res52.npy
    heturun -c config4.yml python test_mlp_mp.py --split 1      --split2 3 --log results/res53.npy
    heturun -c config4.yml python test_mlp_mp.py --split 2      --split2 3 --log results/res54.npy
    heturun -c config4.yml python test_mlp_mp.py --split 3      --split2 3 --log results/res55.npy
    heturun -c config4.yml python test_mlp_mp.py --split 4      --split2 3 --log results/res56.npy
    heturun -c config4.yml python test_mlp_mp.py --split left   --split2 4 --log results/res57.npy
    heturun -c config4.yml python test_mlp_mp.py --split middle --split2 4 --log results/res58.npy
    heturun -c config4.yml python test_mlp_mp.py --split right  --split2 4 --log results/res59.npy
    heturun -c config4.yml python test_mlp_mp.py --split 0      --split2 4 --log results/res60.npy
    heturun -c config4.yml python test_mlp_mp.py --split 1      --split2 4 --log results/res61.npy
    heturun -c config4.yml python test_mlp_mp.py --split 2      --split2 4 --log results/res62.npy
    heturun -c config4.yml python test_mlp_mp.py --split 3      --split2 4 --log results/res63.npy
    heturun -c config4.yml python test_mlp_mp.py --split 4      --split2 4 --log results/res64.npy

    python validate_results.py 65
fi


if $COMPLEX_MP_PP_TEST; then
    heturun -c config6.yml python test_mlp_mp_pp.py --split left   --split2 left --log results/res1.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split middle --split2 left --log results/res2.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split right  --split2 left --log results/res3.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split 0      --split2 left --log results/res4.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split 1      --split2 left --log results/res5.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split 2      --split2 left --log results/res6.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split 3      --split2 left --log results/res7.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split 4      --split2 left --log results/res8.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split left   --split2 right --log results/res9.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split middle --split2 right --log results/res10.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split right  --split2 right --log results/res11.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split 0      --split2 right --log results/res12.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split 1      --split2 right --log results/res13.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split 2      --split2 right --log results/res14.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split 3      --split2 right --log results/res15.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split 4      --split2 right --log results/res16.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split left   --split2 middle --log results/res17.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split middle --split2 middle --log results/res18.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split right  --split2 middle --log results/res19.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split 0      --split2 middle --log results/res20.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split 1      --split2 middle --log results/res21.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split 2      --split2 middle --log results/res22.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split 3      --split2 middle --log results/res23.npy
    heturun -c config6.yml python test_mlp_mp_pp.py --split 4      --split2 middle --log results/res24.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split left   --split2 0 --log results/res25.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split middle --split2 0 --log results/res26.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split right  --split2 0 --log results/res27.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 0      --split2 0 --log results/res28.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 1      --split2 0 --log results/res29.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 2      --split2 0 --log results/res30.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 3      --split2 0 --log results/res31.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 4      --split2 0 --log results/res32.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split left   --split2 1 --log results/res33.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split middle --split2 1 --log results/res34.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split right  --split2 1 --log results/res35.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 0      --split2 1 --log results/res36.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 1      --split2 1 --log results/res37.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 2      --split2 1 --log results/res38.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 3      --split2 1 --log results/res39.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 4      --split2 1 --log results/res40.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split left   --split2 2 --log results/res41.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split middle --split2 2 --log results/res42.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split right  --split2 2 --log results/res43.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 0      --split2 2 --log results/res44.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 1      --split2 2 --log results/res45.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 2      --split2 2 --log results/res46.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 3      --split2 2 --log results/res47.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 4      --split2 2 --log results/res48.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split left   --split2 3 --log results/res49.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split middle --split2 3 --log results/res50.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split right  --split2 3 --log results/res51.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 0      --split2 3 --log results/res52.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 1      --split2 3 --log results/res53.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 2      --split2 3 --log results/res54.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 3      --split2 3 --log results/res55.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 4      --split2 3 --log results/res56.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split left   --split2 4 --log results/res57.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split middle --split2 4 --log results/res58.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split right  --split2 4 --log results/res59.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 0      --split2 4 --log results/res60.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 1      --split2 4 --log results/res61.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 2      --split2 4 --log results/res62.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 3      --split2 4 --log results/res63.npy
    heturun -c config8.yml python test_mlp_mp_pp.py --split 4      --split2 4 --log results/res64.npy

    python validate_results.py 65
fi


if $REMOVE_RESULTS; then
    rm results/*
    rm std/*
fi
