# Codes for the experiments in "[Classification Beats Regression: Counting of Cells from Greyscale Microscopic Images based on Annotation-free Training Samples](https://arxiv.org/pdf/2010.14782.pdf)"

**[NEWS!]** Our paper is accepted by [CAAI International Conference on Artificial Intelligence (CICAI) 2021](https://cicai.caai.cn/#/).

Cell counting on the Tiny-BBBC005 dataset

If you use this code, please cite
```text
@inproceedings{ding2021classification,
  title={Classification Beats Regression: Counting of Cells from Greyscale Microscopic Images Based on Annotation-Free Training Samples},
  author={Ding, Xin and Zhang, Qiong and Welch, William J},
  booktitle={CAAI International Conference on Artificial Intelligence},
  pages={662--673},
  year={2021},
  organization={Springer}
}
```

--------------------------------------------------------

## h5 Datasets
Tiny-BBBC005 <br />
https://1drv.ms/u/s!Arj2pETbYnWQsL0vJF6KJQPIbj_6Fw?e=rsdvw2

VGG Dataset <br />
https://1drv.ms/u/s!Arj2pETbYnWQsL0fsFvYKI5cOt-sxg?e=ceZfFI


--------------------------------------------------------

The following codes are tested on a Ubuntu system with TWO RTX-2080 Ti. <br />

## 1. Ours: ResNet-XX(CE)+DA+Ensemble (in './AfCCNN') <br />
Run scripts in `./AfCCNN/scripts/*.sh`

## 2. DRDCNN (in './DRDCNN') <br />
Liu, Qian, et al. "A novel convolutional regression network for cell counting." 2019 IEEE 7th International Conference on Bioinformatics and Computational Biology (ICBCB). IEEE, 2019. <br />

Run scripts in `./DRDCNN/scripts/*.sh`

## 3. FPNCNN (in './FPNCNN') <br />
Hernández, Carlos X., Mohammad M. Sultan, and Vijay S. Pande. "Using deep learning for segmentation and counting within microscopy data." arXiv preprint arXiv:1802.10548 (2018). <br />

Run scripts in `./FPNCNN/scripts/*.sh`

## 4. ERDCNN (in './ERDCNN') <br />
Liu, Qian, et al. "Automated Counting of Cancer Cells by Ensembling Deep Features." Cells 8.9 (2019): 1019. <br />

Run scripts in `./ERDCNN/scripts/*.sh`

## 5. Regression-oriented ResNets (in './AfCCNN') <br />
Xue, Yao, et al. "Cell counting by regression using convolutional neural network." European Conference on Computer Vision. Springer, Cham, 2016. <br />  
Xue, Yao. "Cell Counting and Detection in Microscopy Images using Deep Neural Network." (2018). <br />

Run scripts in `./AfCCNN/scripts/*.sh`
