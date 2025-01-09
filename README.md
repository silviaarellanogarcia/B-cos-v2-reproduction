# Exploring Interpretability: Reproducing and Testing B-cos Alignment for DNNs

This repository contains the code and experiments for the project "Exploring Interpretability: Reproducing and Testing B-cos Alignment for DNNs", conducted as part of the Advanced Deep Learning course (DD2412) at KTH Royal Institute of Technology. The project implements and extends findings from the paper ["B-cos Alignment for Inherently Interpretable CNNs and Vision Transformers"](https://arxiv.org/abs/2210.00959) focusing on reproducing key experiments and exploring additional novel experiments to assess and expand the capabilities of the B-cos transformation in deep neural networks. Read our main findings [here](https://github.com/user-attachments/files/18360241/Advanced_Deep_Learning___Report_Group_2.1.pdf). 

This work was carried out by [Silvia Arellano](https://github.com/silviaarellanogarcia), [Rosamelia Carioni](https://github.com/rosameliacarioni), and [Arnau Casas](https://github.com/Casassarnau).

--- 

## **Project Overview**
1. **Reproducing results from the paper:**
   - Reproduced experiments to analyze the impact of the B parameter.
   - Evaluated the effect of combining different normalization layers under centered and non-centered configurations.
   - Investigated the trade-off between interpretability and performance.

2. **Evaluation metrics:**
   - Two metrics were used: accuracy and localization with the pointing game metric. 
   - Datesets: CIFAR10 and ImageNet.
     
3. **Novel experiments:**
   - **Adversarial Robustness:** tested the model's interpretability and performance under adversarial noise.
   - **Impact of AddInverse Encoding:** analyzed the role of the AddInverse image encoding technique on B-cos interpretability and performance.
   - **Segmentation Capabilities:** investigated B-cos's ability to handle segmentation tasks using the Pascal VOC dataset.
 

## **Results Overview**

### **Reproducibility of Results:**
- Successfully reproduced key results from the original paper, confirming:
  - The trade-off between interpretability and performance, as controlled by the B parameter, with an optimal balance achieved at B=1.25 (localization: 95.97%, accuracy: 90.29%).
  - Centered and non-centered normalization configurations significantly impacted interpretability, with non-centered LayerNorm providing the best results.
  - Minimal accuracy degradation in pre-trained B-cos models compared to baseline models, demonstrating practical applicability.

<div align="center">
  <img width="518" alt="b_param" src="https://github.com/user-attachments/assets/c6fecd95-c0a4-46fa-afe0-3e91084b3a5b" />
  <img width="729" alt="normalization" src="https://github.com/user-attachments/assets/eaea56a2-4288-49ab-b87b-8ebc6e6e0241" />
  <img width="879" alt="model_comparison" src="https://github.com/user-attachments/assets/a48bd905-9147-49ec-a97d-6f5ac12e2735" />
</div>

### **Adversarial Robustness**
- The B-cos model exhibited reduced performance under adversarial perturbations, with Top-1 accuracy dropping from 67.37% (clean) to 32.80% ($\epsilon=0.5$).
- Explanations became noisier at higher noise levels, indicating a lack of robustness under adversarial attacks.
  
<div align="center">
  <img width="884" alt="adversarial" src="https://github.com/user-attachments/assets/fb2b94d1-1826-4dd2-9d7b-cdb55089bb05" />
</div>

### **Impact of AddInverse Encoding**
- AddInverse encoding marginally improved accuracy (90.29%) and localization scores (95.97%) compared to standard RGB encoding.
- Explanations were notably more detailed, particularly for darker image regions, highlighting its contribution to enhanced interpretability.
  
<div align="center">
  <img width="712" alt="add_inverse" src="https://github.com/user-attachments/assets/8d32f369-af33-4da9-9e31-d133311eed93" />
</div>

### **Segmentation Capabilities**
- The model struggled with segmentation tasks, achieving very low mAP scores ($<0.01$) on Pascal VOC.
- Explanations focused on small object details (e.g., a dog's tongue or a car's wheel) rather than capturing entire objects, revealing limitations in handling segmentation tasks.
  
<div align="center">
  <img width="881" alt="pascal" src="https://github.com/user-attachments/assets/39117b12-88b9-4f98-a3b2-986c12b7d6f0" />
</div>

