# Deep Conflation Model

The Theano code for the ICASSP 2017 paper “[Character-level Deep Conflation for Business Data Analytics](https://arxiv.org/pdf/1702.02640.pdf)” 

## Dependencies

This code is written in python. To use it you will need:

* Python 2.7 (do not use Python 3.0)
* Theano 0.7 (you can also use the most recent version)
* A recent version of NumPy and SciPy 
* Sklearn (This is used for 10-fold cross-validation)

## Getting started

* We do not provide the dataset we used in our paper. However, in the `./data/example_dataset.txt` file, we provide 10 example data samples. Your own dataset should also be formatted in a similar way, and the dataset should be stored in the `./data` folder. 

* We provide two specifications of the dataset: (i) using predefined splits to divide the dataset into training/validation/test set, with the corresponding proportion set to 90%/10%/10%; (ii) implementing 10-fold cross validation. Run `data_preprocess.py` or `data_using_predefined_split.py` for data preprocessing. 

* We also consider two scenarios: (i) using the correct names to query the wrong names; (ii) using the wrong names to query the correct names instead.

## How to use the code

Running the CNN model (which performs the best on this task):

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python eval_rank_cnn.py sys.argv[1] sys.argv[2] 
```

* `sys.argv[1]` takes two values: (i) predefined_split and (ii) cross_validation.
* `sys.argv[2]` also takes two values: (i) normal and (ii) reverse. The former indicates using the correct names to query wrong names, while the latter means using the mis-spelled names to query correct names.

One example:

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python eval_rank_cnn.py predefined_split normal 
```
The CNN model is trained for 20 epochs of the dataset, with mini-batch size set to 100. On a NVIDIA Tesla K40 GPU, the training takes around 45 minutes. However, training for 5 epochs already provides very good results.

The code that implements a LSTM/BoC model can be run in a similar way. All the printout information is stored in one log file.

## Citing DCM

Please cite our ICASSP paper in your publications if it helps your research:

    @inproceedings{DCM_ICASSP2017,
      Author = {Gan, Zhe and Singh, P. D. and Joshi, Ameet and He, Xiaodong and Chen, Jianshu and Gao, Jianfeng and Deng, Li},
      Title = {Character-level Deep Conflation for Business Data Analytics},
      booktitle={ICASSP},
      Year  = {2017}
    }

