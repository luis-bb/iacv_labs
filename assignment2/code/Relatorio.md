# Part 4 — Pretrained visual representations: CNN, DINOv2 and CLIP

## Objective and evaluation protocol

Part 4 evaluates frozen pretrained visual representations for the assignment’s six image categories: aeroplane, background, car, horse, motorbike, and person. ResNet-18, ViT-B/32, and DINOv2 features are used with a linear SVM; CLIP is evaluated separately by comparing image and text embeddings without fitting a task-specific classifier. The available split contains 2,791 training entries and 2,796 validation entries. Because the repository provides no separate test split, the validation split is the final reported evaluation set. SVM hyperparameters are held fixed at `C=1` for the main model comparison.

All deep features are L2-normalized before SVM fitting and evaluation. ResNet-18 uses global average pooling on the outputs of its final three residual blocks (`layer2.1`, `layer3.1`, `layer4.1`), concatenated into an 896-dimensional descriptor. ViT-B/32 uses its 768-dimensional global embedding; DINOv2 ViT-S/14 uses its 384-dimensional CLS embedding. The pretrained backbones remain frozen. CLIP uses ViT-B/32 and the prompt `a photo of a {class}` as the predefined baseline. Its image and text embeddings are compared using cosine similarity.

**Data note.** The supplied category split files overlap: the validation cache contains 2,796 rows but 2,474 unique filenames. The Part 4 classification tables therefore reflect the split entries and their supplied row labels, including repeated images where they appear under multiple category lists. This limits the independence of rows and can make some class decisions ambiguous. The later retrieval experiment aligns and deduplicates features by filename.

## Main results

| Method | Supervision | mAP (five target classes) | Macro-F1 | Feature dimension | Train / validation extraction (s) | SVM fit (s) | Validation scoring (ms/image) | Feature cache (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 + linear SVM | ImageNet pretrained + task labels | 0.7170 | 0.6982 | 896 | 122.37 / 167.39 | 1.3237 | 0.0104 | 17.37 |
| ViT-B/32 + linear SVM | ImageNet pretrained + task labels | 0.7221 | 0.7243 | 768 | 300.87 / 280.13 | 1.2028 | 0.0014 | 15.21 |
| DINOv2 ViT-S/14 + linear SVM | Self-supervised pretrained + task labels | **0.7555** | **0.7647** | 384 | 429.82 / 427.29 | **0.5776** | 0.0019 | **7.61** |
| CLIP zero-shot, `a photo of a {class}` | Image–text pretraining; no task labels | 0.6427 | 0.6229 | 512 | — / 253.63 | — | 0.1598* | 5.07 |

Extraction time includes image loading, preprocessing, and feature inference, but excludes model loading. The scoring column is the notebook’s measured validation scoring time; CLIP includes text-prompt encoding and image–text similarity, while the other methods measure SVM scoring. These numbers are hardware- and implementation-dependent. CLIP’s inference timing is therefore not a strict like-for-like comparison with linear SVM scoring.

| Target class | ResNet-18 AP | ViT-B/32 AP | DINOv2 AP | CLIP baseline AP |
|---|---:|---:|---:|---:|
| Aeroplane | 0.9401 | 0.9268 | 0.9038 | 0.9030 |
| Motorbike | 0.5438 | 0.5717 | 0.6375 | 0.5420 |
| Person | 0.8475 | 0.8241 | 0.8703 | 0.5265 |
| Horse | 0.5095 | 0.5432 | 0.6220 | 0.5677 |
| Car | 0.7440 | 0.7444 | 0.7441 | 0.6742 |

*Figure 4.1 — Insert a bar chart of per-class AP and mAP for ResNet-18, ViT-B/32, DINOv2, and CLIP. The source values are in `part4.ipynb`.*

DINOv2 produced the highest mAP and macro-F1 among the evaluated systems, while using the smallest deep feature vector and cache. It was slower to extract than ResNet-18 and ViT-B/32 in this CPU run. ViT-B/32 marginally exceeded ResNet-18 in mAP (0.7221 versus 0.7170), while ResNet-18’s extraction was faster. These results show that descriptor dimension alone does not predict AP: the smallest embedding performed best. Relative to the Part 1 BoVW result (mAP 0.6037), the three deep SVM systems had higher mAP, but this comparison is descriptive because Part 1 used `C=10` whereas Part 4 uses `C=1`.

### Confusion matrices

Rows are true classes and columns are predicted classes, in the order aeroplane, background, car, horse, motorbike, person.

**DINOv2 + linear SVM**

| True \ Predicted | Aeroplane | Background | Car | Horse | Motorbike | Person |
|---|---:|---:|---:|---:|---:|---:|
| Aeroplane | 126 | 0 | 0 | 0 | 0 | 0 |
| Background | 3 | 1040 | 14 | 0 | 0 | 20 |
| Car | 2 | 25 | 266 | 3 | 6 | 35 |
| Horse | 0 | 7 | 0 | 88 | 0 | 53 |
| Motorbike | 0 | 4 | 4 | 0 | 74 | 43 |
| Person | 13 | 81 | 77 | 61 | 40 | 711 |

**CLIP zero-shot baseline**

| True \ Predicted | Aeroplane | Background | Car | Horse | Motorbike | Person |
|---|---:|---:|---:|---:|---:|---:|
| Aeroplane | 125 | 1 | 0 | 0 | 0 | 0 |
| Background | 9 | 600 | 32 | 137 | 33 | 266 |
| Car | 2 | 77 | 225 | 7 | 21 | 5 |
| Horse | 0 | 3 | 1 | 144 | 0 | 0 |
| Motorbike | 0 | 2 | 0 | 1 | 120 | 2 |
| Person | 14 | 250 | 67 | 136 | 120 | 396 |

The complete ResNet-18 and ViT-B/32 confusion matrices are printed in the notebook. DINOv2 correctly classified 88 of 148 horse entries and 74 of 121 motorbike entries; aeroplane was easiest, with all 126 entries classified correctly. The lower AP and confusion patterns for horse and motorbike indicate that these classes are more difficult in this split. Errors between person, horse, and motorbike are consistent with images containing riders, handlers, or several objects, and with the overlapping image-level class labels; the matrix alone cannot prove the cause of individual errors.

## DINOv2 global features and patch-token pooling

DINOv2’s CLS embedding was compared with the mean of its patch tokens. Both representations are 384-dimensional and use the same training/validation split and `C=1` linear SVM.

| Target class | CLS-token AP | Mean-patch AP |
|---|---:|---:|
| Aeroplane | 0.9038 | 0.9483 |
| Motorbike | 0.6375 | 0.6104 |
| Person | 0.8703 | 0.8805 |
| Horse | 0.6220 | 0.5888 |
| Car | 0.7441 | 0.7539 |
| **mAP** | **0.7555** | **0.7564** |

Replacing the CLS token with the mean patch embedding made almost no difference to overall mAP. It improved aeroplane, person, and car AP, while lowering motorbike and horse AP. Averaging patch tokens removes their explicit spatial locations and much of the arrangement between object parts. Since this experiment averages all tokens into one vector, it does not test methods that retain patch locations or perform region-based retrieval.

*Figure 4.2 — Insert a per-class AP comparison for DINOv2 CLS token versus mean patch tokens.*

## Few-shot learning and horizontal-flip augmentation

The few-shot experiment uses DINOv2 features for binary motorbike retrieval/classification. For each of three random seeds, it samples the stated number of positive motorbike training examples and the same number of negatives from all non-motorbike training entries. Horizontal-flip augmentation adds mirrored versions of the sampled training images. The table reports mean validation AP and standard deviation across seeds.

| Motorbike positives | No augmentation: mean AP ± std | Horizontal flip: mean AP ± std |
|---:|---:|---:|
| 1 | 0.2211 ± 0.2120 | 0.2337 ± 0.2014 |
| 5 | 0.4989 ± 0.0579 | 0.4991 ± 0.0593 |
| 10 | 0.5492 ± 0.0374 | 0.5415 ± 0.0340 |
| 20 | 0.5255 ± 0.0233 | 0.5201 ± 0.0250 |
| 120 | 0.5511 ± 0.0082 | 0.5446 ± 0.0148 |

Performance rises sharply from one to five examples and is close to its all-positive result by ten examples in this setup. The one-example result is highly variable; the curve is not monotonic at 20 examples, so it should not be interpreted as a smooth learning law. Horizontal flipping produced little or no improvement and slightly reduced mean AP at 10, 20, and 120 positives. Flipping can be harmful when left/right orientation, text, or asymmetric context matters, or when a mirrored image is implausible. For a linear classifier, scoring an unnormalized mean of two feature vectors equals averaging their scores; L2-normalizing the mean generally breaks that equality.

*Figure 4.3 — Insert validation AP against the number of motorbike training positives, with separate lines for no augmentation and horizontal flips. Show standard-deviation bars.*

## SVM regularization

The SVM `C` sweep uses five motorbike positives and an equal number of sampled negatives. Training AP is 1.0 for every tested value, while validation AP decreases as `C` grows.

| C | Training AP | Validation AP |
|---:|---:|---:|
| 0.1 | 1.0000 | **0.5610** |
| 1 | 1.0000 | 0.5487 |
| 10 | 1.0000 | 0.5359 |
| 100 | 1.0000 | 0.5327 |
| 1000 | 1.0000 | 0.5323 |

In this small-sample setting, stronger regularization (`C=0.1`) generalizes better among the tested values. The perfect training AP and lower validation AP are consistent with overfitting as `C` increases. However, validation is also the only available evaluation split, so choosing `C` from it makes these results exploratory and not an independent test estimate.

*Figure 4.4 — Insert training and validation AP versus SVM C on a logarithmic C axis.*

## CLIP zero-shot results and prompt sensitivity

The baseline uses CLIP ViT-B/32 with the predefined template `a photo of a {class}`. It obtains mAP 0.6427 and macro-F1 0.6229 over the five target classes. CLIP avoids training a target-specific classifier: it compares each image embedding directly with the embeddings of candidate class descriptions. A linear SVM trained on frozen CLIP image embeddings would instead learn a decision boundary from task labels. The comparison with DINOv2 + SVM (mAP 0.7555, macro-F1 0.7647) is not a controlled comparison of representation alone because DINOv2 uses labeled training data and CLIP zero-shot does not.

| CLIP prompt template | mAP | Macro-F1 |
|---|---:|---:|
| `a photo of a {class}` | 0.6427 | 0.6229 |
| `a picture of a {class}` | **0.6433** | 0.6107 |
| `an image showing a {class}` | 0.6278 | 0.5913 |

Overall mAP changes modestly with these templates, but individual class AP can move more: person AP ranges from 0.4904 to 0.5754, while aeroplane remains around 0.90. The predefined first prompt is retained as the baseline; choosing the top prompt after inspecting validation labels would bias the reported result. Since no test set is available, this is prompt-sensitivity analysis, not an unbiased prompt-selection evaluation.

*Figure 4.5 — Insert CLIP per-class AP by prompt template and the baseline confusion matrix.*

## Visual error analysis and SVM-CAM

The notebook displays DINOv2 false negatives and high-scoring false positives for horse and motorbike. The examples include horse images with people and motorbike images with people or cars; person-labeled entries can also contain prominent animals or vehicles. This is consistent with class co-occurrence and ambiguity in single-label annotations, but a small set of examples cannot establish one cause for all errors.

The ResNet-18 visualization uses the final convolutional feature map and the motorbike class weights of a linear SVM to form a class-specific CAM. In the inspected examples, high positive contributions overlap the motorbike in both a correct example and a motorbike image misclassified as background, with some activation extending to nearby scene context. This map identifies positive contributions to the linear class score; it does not prove causal reliance or explain all information used by the network.

*Figure 4.6 — Insert the notebook’s horse and motorbike false-negative/high-scoring false-positive grids.*

*Figure 4.7 — Insert the ResNet-18 motorbike SVM-CAM examples.*

## Summary and limitations

DINOv2 offers the strongest supervised linear-probe result in this experiment and the most compact deep feature cache. CLIP offers zero-shot text-to-image comparison without task labels, but its class predictions and AP depend on prompt wording. Frozen representations are data-efficient in the motorbike few-shot experiment, while horizontal flips add little under the tested setup. The main limitations are the use of the validation split as the final evaluation, overlapping category split entries, a small number of few-shot seeds, and prompt/hyperparameter comparisons made on the same validation data. The reported timing is specific to the CPU environment and should not be generalized to GPU or other hardware.

### References used in Part 4

- He, K., Zhang, X., Ren, S. and Sun, J. (2016). “Deep Residual Learning for Image Recognition.” *CVPR*.
- Dosovitskiy, A. et al. (2021). “An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.” *ICLR*.
- Oquab, M. et al. (2023). “DINOv2: Learning Robust Visual Features without Supervision.” <https://arxiv.org/abs/2304.07193>.
- Radford, A. et al. (2021). “Learning Transferable Visual Models From Natural Language Supervision.” *ICML*.
- Torchvision pretrained model implementations and weights: <https://pytorch.org/vision/stable/models.html>.
- OpenAI CLIP implementation: <https://github.com/openai/CLIP>.

# Parte 5 - Image Retrieval
