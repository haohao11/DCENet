Exploring  Dynamic  Context  for  Multi-path  Trajectory  Prediction
===


DCENet Structure
===
![DCENet](https://github.com/tanjatang/DCENet/blob/master/pipeline/pipeline.png)


#### Requiements
* python3
* keras-gpu 2.3.1
* tensorflow 2.1.0
* numpy
...

```
pip install -r requiements.txt
```
 
#### Data Preparation
1. download raw data from directory /WORLD H-H TRAJ

2. run /scripts/trainer.py by setting arg.preprocess_data==True for data processing.

**Note:** check if all of the listed directories have already been yieled; set arg.preprocess_data==False if you have already the processed data.

#### Test
You can get the results as reported in the paper using our pretrained model.
1. Download pretrained model from /models/best.hdf5

#### Train
You also can train from sratch by /scripts/trainer.py


#### Citation

If you find our work useful for you, please cite it as:
----
```html
@article{cheng2020exploring,
  title={Exploring Dynamic Context for Multi-path Trajectory Prediction},
  author={Cheng, Hao and Liao, Wentong and Tang, Xuejiao and Yang, Michael Ying and Sester, Monika and Rosenhahn, Bodo},
  journal={arXiv preprint arXiV:2010.16267},
  year={2020}
}
```

#### Update the errors in the inD experiments
First, many credits to Angelos Toytziaridis for helping me discover the evaluation error in the experiments on the inD dataset.
The empirical results have been recalculated and summarized in https://github.com/haohao11/DCENet/tree/master/Extend_inD/results.
The quantitative results may differ significantly from the ones reported in the original papers, but the comparison with the baselines is still valid.
This error does not impact the qualitative results.
If you would like to benchmark your quantitative results to AMENet or DCENet on the inD dataset, you can refer to the updated results or run the experiments (https://github.com/haohao11/DCENet/tree/master/Extend_inD) with the saved weights or train the models yourself.
