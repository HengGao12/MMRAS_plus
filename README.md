# Multi-modality Anomaly Segmentation on the Road
> Heng Gao, Zhuolin He, Shoumeng Qiu, Xiangyang Xue*, Jian Pu  
> Fudan University  


#### Abstract

> Semantic segmentation allows autonomous driving cars to understand the surroundings of the vehicle comprehensively. However, it is also crucial for the model to detect obstacles that may jeopardize the safety of autonomous driving systems. 
> Based on our experiments, we find that current uni-modal anomaly segmentation frameworks tend to produce high anomaly scores for non-anomalous regions in images. Motivated by this empirical finding, we develop a multi-modal **uncertainty-based** anomaly segmentation framework, named MMRAS+, for autonomous driving systems.
> MMRAS+ effectively reduces the high anomaly outputs of non-anomalous classes by introducing text-modal using the CLIP text encoder. Indeed, MMRAS+ is the first multi-modal anomaly segmentation solution for autonomous driving. Moreover, we develop an ensemble module to further boost the anomaly segmentation performance. 
> Experiments on RoadAnomaly, SMIYC, and Fishyscapes validation datasets demonstrate the superior performance of our method.

#### Installation

Please follow the instructions given in [Mask2Anomaly](https://github.com/shyam671/Mask2Anomaly-Unmasking-Anomalies-in-Road-Scene-Segmentation/tree/main).

#### Get Text Embeddings

```python
python tools/prompt_engeering.py
```

The results will be saved in './pretrain/'. We utilize these text embeddings to segment anomalies in road scenes. :)



#### Anomaly Inference

The anomaly inference datasets are given in './datasets/Validation_Dataset', which can be downloaded from this [link](https://drive.google.com/drive/folders/1eQhmPbKSZrN1AsieY9KFchfll7XC1_SF).

```python
CUDA_VISIBLE_DEVICES=1 python anomaly_utils/anomaly_inference.py --score mmras --exp_name the_exp_name
```

The pre-trained model is given in './output/m2unk_coco_supervised_v1/best_contrastive.pth', one can also download the model weights from this [link](https://drive.google.com/drive/folders/1eQhmPbKSZrN1AsieY9KFchfll7XC1_SF).  

**After running the code above, the inference log file will be saved in './results/'**


#### Acknowledgement

Our code is developed based on [Mask2Anomaly](https://github.com/shyam671/Mask2Anomaly-Unmasking-Anomalies-in-Road-Scene-Segmentation/tree/main) and [MaskCLIP](https://github.com/chongzhou96/MaskCLIP/tree/master). Thanks to their great work!

ArXiv Link: Pending update...
