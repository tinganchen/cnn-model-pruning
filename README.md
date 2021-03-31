# CNN pruning - GAL, CVPR 2019 [Reproduce] 
Pruning, model compression, adversarial learning, convolutional neural network.

## References

Towards Optimal Structured CNN Pruning via Generative Adversarial Learning (GAL), CVPR 2019.
* [Paper](https://arxiv.org/abs/1903.09291)
* [Github](https://github.com/ShaohuiLin/GAL) 

FLOP calculation tool- *ptflops*
* [Github](https://github.com/sovrasov/flops-counter.pytorch) 

## Requirements

* python3
* pytorch==1.7.1
* cudatoolkit==11.0.221 
* numpy==1.19.2
* tensorboardx==1.4

## Experiment
ResNet-56 on CIFAR-10. (Image classification)


## Implementation

### Pretrained model preparation

The pretrained weights are downloaded from GAL ([Github](https://github.com/ShaohuiLin/GAL)).


### Training & structure pruning stage

More details of the arguments refer to [options.py](./utils/options.py).

```shell
python main.py --job_dir <experiment_results_dir> --teacher_dir <pretrain_weights_dir> --teacher_file <pretrain_weights_file> --refine None --arch resnet --teacher_model resnet_56 --student_model resnet_56_sparse --num_epochs 100 --train_batch_size 128 --eval_batch_size 100 --lr 0.01 --momentum 0.9 --miu 1 --sparse_lambda 0.6 --lr_decay_step 30 --mask_step 200 --weight_decay 0.0002
```

### Fine-tuning stage

```shell
python finetune.py --job_dir <finetuning_results_dir> --refine <experiment_results_dir> --num_epochs 30 --lr 0.01
```

### Results

Model                | Stage               | #Sructures (blocks)   | FLOPs (pruned ratio)  | #Parameters (pruned ratio) | Top-1 accuracy
---                  |---                  |---                                    |---                    |---                         |---     
Resnet-56 (Original) |Pretrained           | 27                                    |125.49M (0%)           |0.85M (0%)                  | 93.26  
Resnet-56 (Sparse)   |Training & Pruning   | 27                                    |125.49M (0%)           |0.85M (0%)                  | 91.72      
Resnet-56 (Pruned)   |Pruned & Fine-tuning | 17                                    |79.24M (37.7%)         |0.67M (21.7%)               | 92.22       

