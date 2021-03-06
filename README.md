# PU-Transformer-pytorch
Pytorch unofficial implementation of PU-Transformer

(PU-Transformer: Point Cloud Upsampling Transformer)

https://arxiv.org/abs/2111.12242

# Model Structure
**Model**
![ex_screenshot](./img/model.png)


# Evaluation
**Input : 2048**  
**Output : 8192**  
**Test Dataset : PU-GAN dataset**    

| X4 | Chamfer Distance(10<sup>-3</sup>)|HD(10<sup>-3</sup>)|P2F(10<sup>-3</sup>)|
|:--------|:--------:|:--------:|:--------:|
| This code | **0.2718**|**3.161**|**4.202**|
|Paper|**0.273**|**2.605**|**1.836**| 


# Visualize
**Ground Truth**  
![ex_screenshot](./img/cat_gt.png)

**Model Prediction**  
![ex_screenshot](./img/cat_predict.png)

# Acknowledgement
This repo is heavily built on Point Transformer code.

https://github.com/POSTECH-CVLab/point-transformer