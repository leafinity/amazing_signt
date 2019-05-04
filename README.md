# Amazing Signts - Tensorflow 2.0 Implimentation of DCGAN

## abc

## Training

* first time


    python dcgan_fix_512_512.py --train --iteration 1000 --batch_size 8 --data_dir /path/to/your/images


* continue last training (Assuming alreay run 1000 iterations)


    python dcgan_fix_512_512.py --train --iteration 1000 --batch_size 8 --data_dir /path/to/your/images --restore_model --pre_low_step 1000

## Generate 8 Images

    python dcgan_fix_512_512.py --batch_size 8 --result_dir /path/to/your/results

## Results
![result](assets/results.png)

around 8000 mountain images downloaded form Unsplash


