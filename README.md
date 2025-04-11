## 1. Datasets

### CamVid
- **HR Image**: [Download Here](https://github.com/alexgkendall/SegNet-Tutorial)
- **LR Image**: [Download Here](https://drive.google.com/file/d/1Dvd0yNMRmQjsZNKAHNdNDMj8KRShZy83/view?usp=sharing)

### Minicity
- **HR Image**: [Download Here](https://github.com/VIPriors/vipriors-challenges-toolkit/tree/segmentation/semantic-segmentation)
- **LR Image**: [Download Here](https://drive.google.com/file/d/1DAaderRchoBc1uCvu1hHz_VmQZqtn0CG/view?usp=sharing)

---

## 2. Model Weights
<!-- ### CamVid -->
<!-- - **Aset**: [Download Here](https://drive.google.com/drive/folders/1yfWn74q1bGA7SVFDa1tlvbVhS5VaA-b2?usp=sharing) -->
- **Weights**: [Download Here](https://drive.google.com/drive/folders/1wGp8W9yiBPc7OCMOFbUvBhYXKXkVVrGy?usp=sharing)

<!-- ### Minicity -->
<!-- - **Aset**: [Download Here](https://drive.google.com/drive/folders/1ghKGAI1fzbjUB7JE_iWZOqA4z5XsE6Xt?usp=sharing)
- **Bset**: [Download Here](https://drive.google.com/drive/folders/1WRTtidXytTiW8asfqtkxxqqBMDD7p6AF?usp=sharing) -->

---

## 3. Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/2dydgh/KD4SRSS.git
    cd KD4SRSS
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the datasets and weights** by clicking the links provided above.

4. **Run the training script**:
    ```bash
    python main.py
    ```

---


<!-- ## 4. Experimental Results

The following table shows the performance of various semantic segmentation networks on different methods in terms of Pixel Accuracy, Class Accuracy, and Mean Intersection over Union (mIoU):

### CamVid
| Method | Image Type | Pixel Acc | Class Acc | mIoU |
|--------|------------|-----------|-----------|------|
| **DeepLab v3+ [3]** | HR Image | 0.942 | 0.8086 | 0.6913 |
|  | Bilinear interpolation [44] | 0.913 | 0.6266 | 0.5298 |
|  | IMDN [35] | 0.9198 | 0.7242 | 0.5831 |
|  | RFDN [36] | 0.9159 | 0.7088 | 0.571 |
|  | BSRN [37] | 0.9186 | 0.7245 | 0.578 |
|  | ESRT [41] | 0.9183 | 0.723 | 0.576 |
|  | PAN [39] | 0.9198 | 0.7271 | 0.5852 |
|  | LAPAR_A [40] | 0.9195 | 0.7284 | 0.5815 |
|  | ARNet [4] | 0.9315 | 0.7690 | 0.6253 |
|  | **Lightweight SBANet with BAKD (proposed)** | **0.9340** | **0.7819** | **0.6442** |
| **DABNet [2]** | HR Image | 0.9522 | 0.7616 | 0.6834 |
|  | Bilinear interpolation [44] | 0.8942 | 0.5022 | 0.4124 |
|  | IMDN [35] | 0.9047 | 0.5346 | 0.4399 |
|  | RFDN [36] | 0.8998 | 0.5213 | 0.4314 |
|  | BSRN [37] | 0.9004 | 0.5135 | 0.422 |
|  | ESRT [41] | 0.9007 | 0.5218 | 0.4313 |
|  | PAN [39] | 0.9028 | 0.5253 | 0.4323 |
|  | LAPAR_A [40] | 0.9004 | 0.5161 | 0.4303 |
|  | ARNet [4] | 0.9299 | 0.7475 | 0.6082 |
|  | **Lightweight SBANet with BAKD (proposed)** | **0.9285** | **0.7494** | **0.6188** |
| **CGNet [1]** | HR Image | 0.9515 | 0.7587 | 0.6795 |
|  | Bilinear interpolation [44] | 0.8825 | 0.489 | 0.3978 |
|  | IMDN [35] | 0.9019 | 0.5089 | 0.4234 |
|  | RFDN [36] | 0.8991 | 0.5115 | 0.4205 |
|  | BSRN [37] | 0.902 | 0.519 | 0.4361 |
|  | ESRT [41] | 0.9039 | 0.5226 | 0.4397 |
|  | PAN [39] | 0.9026 | 0.5131 | 0.4251 |
|  | LAPAR_A [40] | 0.9023 | 0.5202 | 0.4371 |
|  | ARNet [4] | 0.9316 | 0.7487 | 0.6099 |
|  | **Lightweight SBANet with BAKD (proposed)** | **0.9299** | **0.7534** | **0.6196** |

---
### Minicity

| Method | Image Type | Pixel Acc | Class Acc | mIoU |
|--------|------------|-----------|-----------|------|
| **DeepLab v3+ [3]** | HR Image | 0.9288 | 0.7088 | 0.4378 |
|  | Bilinear interpolation [44] | 0.8776 | 0.5309 | 0.2957 |
|  | IMDN [35] | 0.8926 | 0.5891 | 0.3316 |
|  | RFDN [36] | 0.8824 | 0.5634 | 0.3159 |
|  | BSRN [37] | 0.8919 | 0.5988 | 0.3261 |
|  | ESRT [41] | 0.8786 | 0.5576 | 0.3083 |
|  | PAN [39] | 0.8857 | 0.5929 | 0.3340 |
|  | LAPAR_A [40] | 0.8917 | 0.5849 | 0.3284 |
|  | ARNet [4] | 0.8948 | 0.6138 | 0.3451 |
|  | **Lightweight SBANet with BAKD (proposed)** | **0.8933** | **0.6382** | **0.3493** |
| **DABNet [2]** | HR Image | 0.9356 | 0.6990 | 0.4324 |
|  | Bilinear interpolation [44] | 0.8914 | 0.5840 | 0.3175 |
|  | IMDN [35] | 0.8971 | 0.6031 | 0.3351 |
|  | RFDN [36] | 0.8920 | 0.5869 | 0.3254 |
|  | BSRN [37] | 0.8970 | 0.6019 | 0.3370 |
|  | ESRT [41] | 0.8920 | 0.5893 | 0.3243 |
|  | PAN [39] | 0.8995 | 0.6105 | 0.3415 |
|  | LAPAR_A [40] | 0.8992 | 0.5889 | 0.3349 |
|  | ARNet [4] | 0.8947 | 0.6078 | 0.3417 |
|  | **Lightweight SBANet with BAKD (proposed)** | **0.8911** | **0.6355** | **0.3579** |
| **CGNet [1]** | HR Image | 0.9306 | 0.6833 | 0.4130 |
|  | Bilinear interpolation [44] | 0.8911 | 0.5889 | 0.3194 |
|  | IMDN [35] | 0.9011 | 0.6156 | 0.3406 |
|  | RFDN [36] | 0.8921 | 0.5846 | 0.3227 |
|  | BSRN [37] | 0.8968 | 0.6122 | 0.3341 |
|  | ESRT [41] | 0.8945 | 0.5924 | 0.3261 |
|  | PAN [39] | 0.9039 | 0.6125 | 0.3437 |
|  | LAPAR_A [40] | 0.9019 | 0.6119 | 0.3398 |
|  | ARNet [4] | 0.8998 | 0.6126 | 0.3478 |
|  | **Lightweight SBANet with BAKD (proposed)** | **0.8880** | **0.6394** | **0.3510** | -->
