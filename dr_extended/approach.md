# Materials and Methods

## 1. Overview

Diabetic retinopathy (DR) is a progressive microvascular complication of diabetes mellitus and one of the leading causes of preventable blindness worldwide. Automated grading systems based on fundus photography have the potential to support large-scale screening programmes by providing consistent, reproducible severity assessments. However, deep learning models trained on a single dataset frequently exhibit degraded performance when deployed on images acquired under different clinical conditions, a phenomenon known as domain shift. Furthermore, most existing approaches rely on image-level labels alone, without any mechanism to encourage the model to attend to anatomically meaningful retinal regions.

This work addresses both limitations through a dual-resolution deep learning framework for five-class DR severity grading (grades 0–4). The framework combines two parallel EfficientNet backbones operating at different spatial resolutions with a learnable attention-gated fusion module and an ordinal regression head based on the CORAL (Cumulative Ordinal Regression with Logistic loss) formulation. A self-supervised attention consistency loss is introduced as a novel training signal: by enforcing that the model's spatial attention maps are equivariant under horizontal flipping, the model is implicitly guided toward genuine retinal lesion regions without requiring pixel-level annotations. At inference time, an ensemble of two independently trained models — a baseline model trained with CORAL loss only, and an attention-consistency model trained with the combined objective — is evaluated using both standard averaging and test-time augmentation (TTA) via horizontal flipping, further improving cross-dataset robustness.

---

## 2. Datasets and Preprocessing

### 2.1 Datasets

Two publicly available retinal fundus image collections were used for training and internal validation, and a third independent dataset was used exclusively for external evaluation. The **APTOS 2019** dataset comprises 3,662 training images labelled with DR severity grades on a five-point ordinal scale: 0 (no DR), 1 (mild), 2 (moderate), 3 (severe), and 4 (proliferative DR). The **Messidor-2** dataset contains fundus images collected from multiple clinical centres using heterogeneous acquisition devices, resulting in substantial variability in illumination, colour distribution, and field of view. Each Messidor-2 image is labelled with a DR severity grade using the same five-level ordinal scale (grades 0–4); however, grade 3 (severe DR) is absent from the Messidor-2 test split used in this study, as confirmed by the all-zero grade-3 row in the corresponding confusion matrices. Only images marked as both adjudicated and gradable were retained to ensure label reliability.

The **IDRiD (Indian Diabetic Retinopathy Image Dataset) Disease Grading** subset was used as a second independent external test set. It comprises 103 test images with ground-truth DR severity grades on the same five-level ordinal scale (grades 0–4), collected from a single clinical centre in India using a Kowa VX-10α fundus camera. IDRiD was not used at any point during training or model selection; it was evaluated solely to assess the generalisability of the final ensemble pipeline to a dataset from a different geographic region and acquisition device.

### 2.2 Data Splits

The APTOS 2019 dataset was partitioned using stratified five-fold cross-validation (random seed 42). The first fold was designated as the internal validation set; the remaining four folds formed the APTOS training portion. This yielded approximately 2,929 training images and 733 validation images from APTOS.

The filtered Messidor-2 images were split into a training portion (70%) and a held-out external test set (30%) using stratified random splitting (random seed 42), yielding approximately 840 training images and 360 test images. The final combined training set consisted of all APTOS training images together with the Messidor-2 training portion (approximately 3,769 images in total). The external Messidor-2 test set was never used during training or model selection and was reserved exclusively for final evaluation.

The IDRiD Disease Grading test set (103 images) was used in its entirety as a second external evaluation set. No IDRiD images were included in training or validation.

### 2.3 Preprocessing Pipeline

Retinal fundus images exhibit substantial variation in background illumination and field-of-view extent. A multi-stage preprocessing pipeline was applied to all images to reduce this variability.

**Retinal region isolation.** Each image was circularly cropped to remove the uninformative dark background surrounding the retinal disc. The crop was computed using OTSU thresholding on the green channel to produce a binary mask, from which the largest contour was extracted. The minimum enclosing circle of this contour defined the crop region, which was then padded to a square and resized. This step ensures a consistent spatial layout across images from different sources.

**Histogram matching for domain alignment.** To reduce the colour and intensity discrepancy between APTOS and Messidor-2 images, histogram matching was applied to all Messidor-2 images. A reference image was constructed by averaging the colour distributions of up to 50 evenly-spaced APTOS training images after circular cropping and resizing to $512 \times 512$ pixels. Each Messidor-2 image was then adjusted to match this mean reference using channel-wise cumulative distribution function (CDF) matching. This step was applied only to Messidor-2 images; APTOS images were used without histogram matching.

**Dual-branch image enhancement.** Two complementary enhancement strategies were applied in parallel to produce the inputs for the two network branches:

- *Branch 0 — Ben Graham normalisation:* The cropped image was resized to $224 \times 224$ pixels. A Gaussian-blurred version of the image (kernel size proportional to image width, $\sigma = 0.1 \times \text{width}$) was subtracted from the original, and a constant of 128 was added to restore the mean intensity. Pixel values were clipped to $[0, 255]$ and normalised to $[0, 1]$. This operation suppresses low-frequency illumination gradients and enhances blood vessel contrast.

- *Branch 3 — CLAHE enhancement:* The cropped image was resized to $300 \times 300$ pixels. Contrast-Limited Adaptive Histogram Equalisation (CLAHE) was applied to the L channel of the LAB colour space with a clip limit of 2.0 and a tile grid size of $8 \times 8$. The enhanced L channel was recombined with the original A and B channels and converted back to RGB. This operation boosts local contrast and improves the visibility of small lesions such as microaneurysms and exudates.

Both enhanced images were subsequently normalised using ImageNet channel-wise mean $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$ and standard deviation $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$.

**Offline preprocessing.** To accelerate training, the circular crop and histogram matching steps were performed offline and the resulting base images were stored as compressed NumPy arrays (`.npy` files). The branch-specific enhancements (Ben Graham and CLAHE) and augmentations were applied on-the-fly during data loading.

**Training augmentation.** During training only, the following stochastic augmentations were applied to the base image before branch-specific enhancement: random horizontal flipping (probability 0.5), random brightness and contrast adjustment (limit $\pm 0.2$, probability 0.5), and random gamma correction (gamma range $[80, 120]$, probability 0.3). No augmentation was applied during validation or testing.

**Class balancing.** To address the substantial class imbalance in the combined training set, a weighted random sampler was used during training. The sampling weight for each training example was set to $w_i = 1 / (K \cdot n_{y_i})$, where $K = 5$ is the number of classes and $n_{y_i}$ is the number of training examples with label $y_i$.

---

## 3. Baseline Architecture

### 3.1 Dual-Branch Feature Extraction

The model employs two EfficientNet backbones operating in parallel. The first branch uses **EfficientNet-B0** (pretrained on ImageNet, final classification layer removed), which accepts $224 \times 224$ Ben-Graham-normalised images and produces a 1280-dimensional pooled feature vector $\mathbf{f}_0 \in \mathbb{R}^{1280}$. The second branch uses **EfficientNet-B3** (pretrained on ImageNet, final classification layer removed), which accepts $300 \times 300$ CLAHE-enhanced images and produces a 1536-dimensional pooled feature vector $\mathbf{f}_3 \in \mathbb{R}^{1536}$. The two branches capture complementary information: the B0 branch emphasises global retinal structure, while the B3 branch captures finer lesion details at higher resolution.

### 3.2 Branch Projection

Both feature vectors were projected to a common fusion dimension $d_\text{fusion} = 1536$ via separate linear layers:

$$\tilde{\mathbf{f}}_0 = \mathbf{W}_0 \mathbf{f}_0 + \mathbf{b}_0, \quad \tilde{\mathbf{f}}_3 = \mathbf{W}_3 \mathbf{f}_3 + \mathbf{b}_3$$

where $\mathbf{W}_0 \in \mathbb{R}^{1536 \times 1280}$ and $\mathbf{W}_3 \in \mathbb{R}^{1536 \times 1536}$. This symmetric projection ensures that both branches contribute equally to the fusion space and decouples the fusion dimension from the native backbone output dimensions.

### 3.3 Attention-Gated Feature Fusion

The projected feature vectors were fused using a squeeze-and-excitation-style attention gate. The two vectors were concatenated to form $\mathbf{f}_\text{concat} = [\tilde{\mathbf{f}}_0 \| \tilde{\mathbf{f}}_3] \in \mathbb{R}^{2 \times d_\text{fusion}}$, which was passed through a two-layer MLP with a sigmoid activation to produce a channel-wise gating vector:

$$\mathbf{g} = \sigma\!\left(\mathbf{W}_2 \cdot \text{ReLU}\!\left(\mathbf{W}_1 \mathbf{f}_\text{concat} + \mathbf{b}_1\right) + \mathbf{b}_2\right) \in (0,1)^{d_\text{fusion}}$$

The fused feature vector was then computed as a convex combination of the two projected branch features:

$$\mathbf{f}_\text{fused} = \mathbf{g} \odot \tilde{\mathbf{f}}_0 + (1 - \mathbf{g}) \odot \tilde{\mathbf{f}}_3$$

This formulation allows the model to adaptively weight the contribution of each resolution branch on a per-channel, per-sample basis.

### 3.4 CORAL Ordinal Regression Head

The fused feature vector $\mathbf{f}_\text{fused}$ was passed to a CORAL ordinal regression head consisting of a single linear layer mapping $d_\text{fusion} = 1536$ to $K - 1 = 4$ logits, where $K = 5$ is the number of DR severity grades. Each logit $z_k$ represents the log-odds that the true grade exceeds threshold $k$.

The predicted DR grade was obtained by thresholding the sigmoid-transformed logits:

$$\hat{y} = \sum_{k=0}^{K-2} \mathbf{1}\!\left[\sigma(z_k) > 0.5\right]$$

The CORAL loss was computed as the mean binary cross-entropy between the predicted logits and the rank-consistent binary targets $t_k = \mathbf{1}[y > k]$:

$$\mathcal{L}_\text{CORAL} = \frac{1}{B(K-1)} \sum_{i=1}^{B} \sum_{k=0}^{K-2} \left[ -t_k^{(i)} \log \sigma(z_k^{(i)}) - (1 - t_k^{(i)}) \log (1 - \sigma(z_k^{(i)})) \right]$$

where $B$ is the batch size. This formulation explicitly encodes the ordinal structure of DR severity: a prediction error between adjacent grades is penalised less than an error spanning multiple grades.

---

## 4. Self-Supervised Attention Consistency (Novel Contribution)

### 4.1 Motivation

The attention-gated fusion module implicitly learns to weight spatial features, but there is no explicit supervision to ensure that the model attends to genuine retinal pathology. Pixel-level lesion annotations are expensive to obtain and were not available for the datasets used in this study. To address this, a self-supervised attention consistency loss was introduced that exploits the approximate left-right symmetry of fundus images. Specifically, the spatial attention map produced by the model for a horizontally flipped image should be the horizontal mirror of the attention map produced for the original image. Enforcing this equivariance encourages the model to focus on anatomically meaningful structures rather than dataset-specific artefacts.

### 4.2 Feature Map Extraction

Forward hooks were registered on the `conv_head` layer of each EfficientNet backbone — the final convolutional layer before global average pooling. During each forward pass, these hooks captured the intermediate feature maps:

- B0 branch: $\mathbf{F}_0 \in \mathbb{R}^{B \times 1280 \times 7 \times 7}$ (at $224 \times 224$ input resolution)
- B3 branch: $\mathbf{F}_3 \in \mathbb{R}^{B \times 1536 \times 10 \times 10}$ (at $300 \times 300$ input resolution)

### 4.3 Attention Map Computation

A spatial saliency map was derived from each feature map by computing the $\ell_2$ norm across the channel dimension:

$$\mathbf{A}_i = \|\mathbf{F}_i\|_2 \in \mathbb{R}^{B \times 1 \times H_i \times W_i}, \quad i \in \{0, 3\}$$

Each map was then min-max normalised per sample to the range $[0, 1]$:

$$\hat{\mathbf{A}}_i = \frac{\mathbf{A}_i - \min(\mathbf{A}_i)}{\max(\mathbf{A}_i) + \epsilon}$$

where $\epsilon = 10^{-8}$ prevents division by zero. Both normalised maps were bilinearly upsampled to a common spatial resolution of $224 \times 224$ and averaged to produce a single fused attention map:

$$\mathbf{A}_\text{fused} = \frac{\hat{\mathbf{A}}_0^\uparrow + \hat{\mathbf{A}}_3^\uparrow}{2}$$

### 4.4 Consistency Loss

For each training batch, the fused attention map $\mathbf{A}_\text{fused}$ was computed from the feature maps produced by the primary forward pass. A second forward pass was then performed on horizontally flipped versions of the input images $(\mathbf{x}_0^\text{flip}, \mathbf{x}_3^\text{flip})$ under `torch.no_grad()` to obtain the attention map of the flipped inputs, $\mathbf{A}_\text{flip}$. The attention consistency loss was defined as the mean squared error between the horizontally flipped primary attention map and the attention map of the flipped inputs:

$$\mathcal{L}_\text{att} = \text{MSE}\!\left(\text{flip}(\mathbf{A}_\text{fused}),\ \mathbf{A}_\text{flip}\right)$$

This loss penalises asymmetries in the model's spatial attention that cannot be explained by genuine retinal anatomy.

---

## 5. Training Strategy

### 5.1 Combined Objective

The attention-consistency model was trained by minimising the following combined loss:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{CORAL} + \lambda_\text{att} \cdot \mathcal{L}_\text{att}$$

where $\lambda_\text{att} = 0.1$ was the attention consistency weight. The baseline model was trained using $\mathcal{L}_\text{CORAL}$ alone (i.e., $\lambda_\text{att} = 0$).

### 5.2 Optimisation

Both models were trained using the Adam optimiser with a learning rate of $1 \times 10^{-4}$ and weight decay of $0$. Mixed-precision training was enabled using PyTorch's automatic mixed precision (AMP) with a `GradScaler` to prevent gradient underflow. Training was conducted for a maximum of 30 epochs with a batch size of 16. A weighted random sampler was used to balance class frequencies in each training batch, as described in Section 2.3.

### 5.3 Model Selection and Early Stopping

The quadratic weighted kappa (QWK) on the APTOS validation set was monitored after each epoch. The checkpoint achieving the highest validation QWK was retained. Training was terminated early if no improvement in validation QWK was observed for 5 consecutive epochs (patience = 5), after which the best checkpoint was restored for evaluation.

### 5.4 Domain Adaptation Variant

An optional domain adversarial training mode (DANN) was also implemented, incorporating a gradient reversal layer and a domain classifier head. This mode was controlled via the configuration flag `domain_adaptation.enabled`. In the experiments reported here, domain adaptation was disabled (`domain_adaptation.enabled = false`), and the model was trained in the standard combined-loader mode using all labelled training data.

---

## 6. Ensemble Inference

Two independently trained models were combined at inference time: (1) the **baseline model**, trained with CORAL loss only on the combined APTOS + Messidor-2 training set, and (2) the **attention-consistency model**, trained with the combined CORAL and attention consistency objective. Both models share the same architecture (dual-branch EfficientNet with attention-gated fusion and CORAL head, $d_\text{fusion} = 1536$, $K = 5$).

For each test image, the raw CORAL logits from both models were combined using two fusion strategies:

**Logit averaging:** The logit vectors from both models were averaged element-wise before applying the sigmoid function and the threshold-sum decoding:

$$\hat{y}_\text{logit} = \sum_{k=0}^{K-2} \mathbf{1}\!\left[\sigma\!\left(\frac{z_k^\text{base} + z_k^\text{attn}}{2}\right) > 0.5\right]$$

**Probability averaging:** The sigmoid function was applied to each model's logits independently, and the resulting probability vectors were averaged before thresholding:

$$\hat{y}_\text{prob} = \sum_{k=0}^{K-2} \mathbf{1}\!\left[\frac{\sigma(z_k^\text{base}) + \sigma(z_k^\text{attn})}{2} > 0.5\right]$$

Due to the monotonicity of the sigmoid function, both strategies produce identical predictions when the two models agree on the sign of each logit. Differences arise only near the decision boundary, where the two formulations may diverge slightly.

---

## 7. Test-Time Augmentation (TTA)

To further improve cross-dataset generalisation without retraining, test-time augmentation via horizontal flipping was applied during inference. For each test image, both models were evaluated on the original image and on its horizontally flipped version, yielding four logit vectors per sample: $z^\text{base}_\text{orig}$, $z^\text{base}_\text{flip}$, $z^\text{attn}_\text{orig}$, and $z^\text{attn}_\text{flip}$.

**Logit averaging with TTA:** All four logit vectors were averaged element-wise before sigmoid and threshold-sum decoding:

$$\hat{y}_\text{TTA-logit} = \sum_{k=0}^{K-2} \mathbf{1}\!\left[\sigma\!\left(\frac{z_k^\text{base,orig} + z_k^\text{base,flip} + z_k^\text{attn,orig} + z_k^\text{attn,flip}}{4}\right) > 0.5\right]$$

**Probability averaging with TTA:** The sigmoid function was applied to each of the four logit vectors independently, and the four probability vectors were averaged before thresholding:

$$\hat{y}_\text{TTA-prob} = \sum_{k=0}^{K-2} \mathbf{1}\!\left[\frac{\sigma(z_k^\text{base,orig}) + \sigma(z_k^\text{base,flip}) + \sigma(z_k^\text{attn,orig}) + \sigma(z_k^\text{attn,flip})}{4} > 0.5\right]$$

This procedure exploits the approximate bilateral symmetry of fundus images: the retinal vasculature and lesion distribution are broadly symmetric about the vertical axis, so averaging predictions over both orientations reduces the influence of random asymmetric artefacts introduced during image acquisition or preprocessing.

---

## 8. Evaluation Metrics

Model performance was assessed using two metrics. The primary metric was the **Quadratic Weighted Kappa (QWK)**, defined as:

$$\kappa = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}$$

where $O_{ij}$ is the observed count of samples with true grade $i$ and predicted grade $j$, $E_{ij}$ is the expected count under the assumption of independence between true and predicted grades, and $w_{ij} = (i - j)^2 / (K-1)^2$ is the quadratic penalty weight. QWK is particularly appropriate for ordinal grading tasks because it penalises large disagreements more heavily than small ones, reflecting the clinical significance of misclassifying a severe case as mild.

The secondary metric was the **5×5 confusion matrix**, which provides a detailed breakdown of per-class prediction accuracy and the distribution of misclassification errors across severity levels.

All metrics were computed on the external test sets — the held-out Messidor-2 portion and the IDRiD Disease Grading test set — using the final ensemble and TTA pipeline described in Sections 6 and 7. Neither external test set was used at any point during training or model selection.


---

## 9. Experimental Results

### 9.1 Internal Validation — APTOS 2019 (733 images)

All model variants were evaluated on the held-out APTOS validation fold. QWK values remained stable across configurations, confirming that the extensions did not degrade in-domain performance.

| Configuration | QWK |
|---|---|
| Baseline (CORAL only) | 0.9042 |
| DANN + attention consistency (collapsed) | 0.9133 |
| Attention consistency (single model) | 0.9110 |
| DANN + semi-supervised | 0.9016 |

All values fall within [0.90, 0.91], indicating that the attention consistency loss and domain adaptation variants preserved in-domain grading accuracy.

The confusion matrix for the attention-consistency model on the APTOS validation set (rows = true grade, columns = predicted grade, grades 0–4) was:

```
[[360,   0,   0,   0,   1],
 [  2,  34,  37,   0,   1],
 [  0,  15, 160,  20,   5],
 [  0,   0,  10,  23,   6],
 [  0,   0,  17,  12,  30]]
```

Misclassifications were predominantly between adjacent severity grades, consistent with the ordinal structure enforced by the CORAL loss.

---

### 9.2 External Evaluation — Messidor-2 Test Set (360 images)

The Messidor-2 held-out test set was used to assess cross-dataset generalisation. Results are reported for all configurations evaluated.

| Configuration | QWK |
|---|---|
| Baseline | 0.6743 |
| DANN + semi-supervised | 0.6227 |
| DANN + attention (collapsed) | 0.3012 |
| Attention consistency (single model) | 0.7035 |
| Ensemble (baseline + attention) | 0.7125 |
| **Ensemble + TTA (best)** | **0.7329** |

*The ensemble with TTA improved over the baseline by +8.7 % (0.6743 → 0.7329).*

Confusion matrices (rows = true grade, columns = predicted grade, grades 0–4):

**Baseline (QWK = 0.6743)**
```
[[113,  45,   6,   0,   0],
 [ 17,  22,   6,   0,   1],
 [ 25,  17,  18,   2,  12],
 [  0,   0,   0,   0,   0],
 [  5,  11,  17,   2,  41]]
```

**Attention consistency — single model (QWK = 0.7035)**
```
[[ 97,  49,  17,   1,   0],
 [ 15,  16,  14,   0,   1],
 [ 12,  17,  31,   1,  13],
 [  0,   0,   0,   0,   0],
 [  3,   9,  14,   3,  47]]
```

**Ensemble — baseline + attention (QWK = 0.7125)**
```
[[106,  49,   9,   0,   0],
 [ 17,  20,   8,   0,   1],
 [ 16,  20,  25,   2,  11],
 [  0,   0,   0,   0,   0],
 [  4,   6,  21,   1,  44]]
```

**Ensemble + TTA — best (QWK = 0.7329)**
```
[[115,  42,   7,   0,   0],
 [ 18,  20,   7,   0,   1],
 [ 20,  19,  25,   1,   9],
 [  0,   0,   0,   0,   0],
 [  3,   6,  20,   2,  45]]
```

The grade-3 row is all zeros across all configurations, reflecting the absence of grade-3 (severe DR) samples in the Messidor-2 test split. This is a known characteristic of the Messidor-2 label distribution after adjudicated/gradable filtering, not a model failure.

---

### 9.3 External Evaluation — IDRiD Disease Grading Test Set (103 images)

The IDRiD Disease Grading test set provided a second independent external evaluation, assessing generalisation to a dataset not seen during training.

| Configuration | QWK |
|---|---|
| Baseline | 0.5638 |
| Attention consistency (single model) | 0.5719 |
| Ensemble (baseline + attention) | 0.5814 |
| **Ensemble + TTA (best)** | **0.6180** |

*The ensemble with TTA improved over the baseline by +9.6 % (0.5638 → 0.6180).*

Confusion matrices (rows = true grade, columns = predicted grade, grades 0–4):

**Baseline (QWK = 0.5638)**
```
[[29,   3,   2,   0,   0],
 [ 4,   1,   0,   0,   0],
 [11,   2,  17,   1,   1],
 [ 2,   2,  11,   4,   0],
 [ 1,   2,   6,   3,   1]]
```

**Attention consistency — single model (QWK = 0.5719)**
```
[[20,   2,   9,   2,   1],
 [ 4,   0,   1,   0,   0],
 [ 7,   0,   9,  10,   6],
 [ 2,   0,   2,  10,   5],
 [ 0,   1,   1,   8,   3]]
```

**Ensemble — baseline + attention (QWK = 0.5814)**
```
[[25,   2,   7,   0,   0],
 [ 5,   0,   0,   0,   0],
 [ 7,   3,  16,   3,   3],
 [ 2,   0,   9,   7,   1],
 [ 1,   1,   5,   5,   1]]
```

**Ensemble + TTA — best (QWK = 0.6180)**
```
[[25,   3,   6,   0,   0],
 [ 4,   1,   0,   0,   0],
 [ 8,   2,  15,   5,   2],
 [ 2,   0,   8,   7,   2],
 [ 1,   1,   3,   6,   2]]
```

---

### 9.4 Summary of Best Results

| Dataset | Baseline QWK | Best Method | Best QWK | Improvement |
|---|---|---|---|---|
| APTOS 2019 (val, 733 images) | 0.9042 | Attention consistency | 0.9110 | +0.75 % (negligible) |
| Messidor-2 (ext. test, 360 images) | 0.6743 | Ensemble + TTA | 0.7329 | +8.7 % |
| IDRiD (ext. test, 103 images) | 0.5638 | Ensemble + TTA | 0.6180 | +9.6 % |

The ensemble with TTA consistently outperformed all single-model configurations on both external test sets, demonstrating that combining complementary model predictions with horizontal-flip augmentation at inference time is an effective and training-free strategy for improving cross-dataset DR grading robustness.
 The DANN variant with collapsed domain labels (QWK = 0.3012) exhibited severe degradation, indicating that aggressive domain adversarial training without careful hyperparameter tuning is detrimental to grading performance on this dataset.