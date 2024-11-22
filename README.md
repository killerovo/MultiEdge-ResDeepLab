# MultiEdge-ResDeepLab
A Deep Learning Architecture for Multi-Modal Medical Image Segmentation with Enhanced Edge Detection

这段代码实现了一个基于 ResNet 的 DeepLab v3+ 模型，主要用于图像分割任务。它结合了原始的图像信息与边缘信息，通过提取图像的边缘特征来增强模型对图像细节的学习能力。以下是代码的中文介绍：

### 1. **边缘提取函数（`extract_edges_gpu`）**  
此函数使用 Sobel 算子对输入图像进行边缘检测。具体地，通过分别应用水平和垂直方向的 Sobel 滤波器，计算图像中每个像素的梯度大小，并返回一个表示图像边缘的张量。该操作利用了 GPU 加速，以提高性能。

### 2. **模态图像拼接（`concatenate_modalities`）**  
该函数将三个不同模态的医学图像（如 T1、T1CE 和 T2）按通道维度进行拼接，生成一个融合了多模态信息的输入张量。

### 3. **边缘与模态图像拼接（`concatenate_edge_and_modalities`）**  
此函数首先通过 `extract_edges_gpu` 提取每个模态图像的边缘特征，然后将原始图像和相应的边缘图像拼接在一起，生成一个包含边缘信息的多通道输入。拼接后的边缘信息可以帮助模型更好地关注图像中的细节区域。

### 4. **卷积层（`conv1x1`, `conv3x3`, `conv5x5`, `atrous_conv`）**  
这些函数定义了不同类型的卷积操作，包括标准的 1x1、3x3、5x5 卷积，以及膨胀卷积（dilated convolution）。膨胀卷积特别用于捕获更大的感受野，有助于处理大尺度的图像结构。

### 5. **ResNet 架构（`ResNet` 和相关模块）**  
ResNet 架构是 DeepLab v3+ 的骨干网络，用于提取图像特征。代码中实现了基于 `BasicBlock` 和 `Bottleneck` 的 ResNet 模型。通过多层的卷积和残差连接，ResNet 能够有效地训练深层网络，避免梯度消失的问题。

### 6. **DeepLab v3+ 模型（`Deeplab_V3_Plus`）**  
这是 DeepLab v3+ 网络的核心实现，结合了多个模块来进行图像分割。具体流程如下：
   - **ResNet Backbone**：通过 ResNet 提取图像的高级特征。
   - **边缘信息提取与融合**：提取输入图像和边缘图像的特征，并将它们融合，增强模型对细节的关注。
   - **Encoder**：使用膨胀卷积和池化操作进行编码，捕获不同尺度的特征。
   - **分类头（`cls_head`）**：用于将最终的特征映射转换为类别预测。

### 7. **预测与损失计算**  
在 `forward` 函数中，输入图像经过 ResNet 提取特征后，融合边缘信息进行更精准的图像分割预测。最终的输出通过上采样恢复到原图尺寸，得到每个像素的类别概率分布。

### 总结  
该代码实现了一个基于 ResNet 的 DeepLab v3+ 图像分割网络，结合了多模态图像和边缘信息，在图像分割任务中具有更强的表现。通过利用 ResNet 强大的特征提取能力和边缘信息的辅助，模型能够更好地捕捉到图像中的细节，提高分割精度。
