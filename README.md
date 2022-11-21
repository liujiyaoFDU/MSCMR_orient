# Cardiac MRI Orientation Adjust Tool



In this paper, the problem of orientation correction in cardiac MRI images is investigated and a framework for orientation recognition via deep neural networks is proposed. For  multi-modality MRI, we introduce a transfer learning strategy to transfer our proposed model from single modality to multi-modality. We embed the proposed network into the orientation correction command-line tool, which can implement orientation correction on 2D DICOM and 3D NIFTI images. 

![](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/iWQePA.png)

## preprocess

+ **truncation and concat**:
  + For the maximum pixel value of each 2D slice as $X_{max}$,  truncation operations are performed on $X_t$ at the threshold of 60%, 80% and 100% of $X_{max}$ to obtain $X_{1t},X_{2t},X_{3t}$.
  + Concatenated 3-channel image $[X_{1t}, X_{2t}, X_{3t}]$ as $X'$
+ **histogram equalization**;
+ **random small-angle rotations, random crops (in training), and resize;**
+ **z-score normalization**

## training

![image-20221121191527760](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/image-20221121191527760.png)

we pre-train the model on the bSSFP cine dataset, and then transfer the model to the late gadolinium enhancement (LGE) CMR or T2-weighted CMR dataset. On the new modality dataset, we first load the pre-trained model parameters. We freeze the network parameters of the backbone and retrain the fully connected layers on the new modality dataset.

## result

![image-20221121191545900](https://ossjiyaoliu.oss-cn-beijing.aliyuncs.com/uPic/image-20221121191545900.png)



