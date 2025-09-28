# Vision Transformer (ViT) Replication - FoodVision Mini 🍕🥩🍣

This project replicates the original [ViT paper](https://arxiv.org/abs/2010.11929), 
*"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*, 
using PyTorch and applies it to the FoodVision Mini dataset (pizza, steak, sushi).

## Project Structure
- `notebooks/` → step-by-step Jupyter notebook replication
- `src/` → modular PyTorch code for data setup, model, training, and utils
- `experiments/` → results, logs, trained models

## Notes - 

####
1) **Data Loaders and Setup** - Start out with image of (224, 224, 3) -> (height, width, colour_channels)

2) **Patch and Positional Embedding**
    - We want a smaller "patch" of image to be processed parallelly - we choose patch_size = 16
    - Expected output after patch embedding -> 2D tensor of (196, 768) -> ((14)^2, (16)^2 * 3) -> (number_of_patches, embedding_dimension = (patch_size)^2 * colour_channels)
    - Then run each patch through a Conv2D layer (kernel_size = patch_size)(stride = patch_size) -> creating a "feature map" of each patch = embedding layer 
    - output of this layer -> (batch_size, embedding_dimension, height_feature_map, width_feature_map) -> (1, 768, 14, 14)
    - Need to flatten (the spatial dimensions(14x14)) to a 1D sequence of flattened 2D patches - Linear layer of 196 patches (1, 768, 196) -> Patch embeddings!
    - Transpose Patch Embeddings to get sequence-first format - (1, 196. 768) 
    - These are trainable but we need a "classifier prediction" at the end so we also prepend a CLS token(basically) to the patch embedding layer - CLS: (1, 1, 768); patch_embeddings: (1, 196, 768)
    - Now, the Patch embedding + CLS layer looks like [CLS_token_embedding, patch_embeddings] = total size (number of patches + 1) (1, 197, 768)
    - Finally, adding a positional embedding to each patch (to know where it was in the image originally), create a tensor of the same shape as the Patch embedding + CLS layer (1, 197, 768) and just add to the patch embeddings layer
    
3) **Multi-Head Self-Attention (MSA)**
    - The core of transformer architecture - allows each patch to "attend" to other patches
    - Takes input of shape (1, 197, 768) and maintains same output shape
    - Multiple attention heads (12 heads for ViT-Base) run in parallel, each learning different relationships
    - Each head has embedding_dim // num_heads = 768 // 12 = 64 dimensions
    - **LayerNorm** applied BEFORE the MSA block (Pre-Norm) - normalizes across the embedding dimension
    - **Residual (skip) connections** - adds input directly to output of MSA block to help with gradient flow
    - Formula: `output = LayerNorm(input) -> MSA -> + input` (residual connection)

4) **MLP Block (Feed-Forward Network)**
    - Simple 2-layer neural network applied to each patch embedding independently 
    - Takes (1, 197, 768) -> expands to (1, 197, 3072) -> back to (1, 197, 768)
    - Uses GELU activation function (smoother than ReLU)
    - Includes dropout for regularization (0.1 in ViT-Base)
    - **LayerNorm** applied BEFORE MLP block (Pre-Norm)
    - **Residual connection** again: `output = LayerNorm(input) -> MLP -> + input`

5) **Transformer Encoder Block**
    - Combines MSA + MLP blocks in sequence with their respective LayerNorms and residual connections
    - Complete block: `x -> LayerNorm -> MSA -> +x -> LayerNorm -> MLP -> +x`
    - This is the fundamental building block - ViT-Base uses 12 of these blocks stacked
    - Input/Output shape stays (1, 197, 768) throughout all blocks

6) **Full ViT Architecture**
    - **Patch Embedding** (with class token + positional embedding) -> (1, 197, 768)
    - **Stack of Transformer Encoder Blocks** (12 blocks for ViT-Base) -> (1, 197, 768)
    - **Classification Head** - takes only the CLS token (first token) -> (1, 768) -> Linear layer -> (1, num_classes)
    - The CLS token accumulates global information from all patches through self-attention

7) **Training Process**
    - **Custom ViT**: Train from scratch with Adam optimizer (lr=3e-3, weight_decay=0.3)
    - **Loss Function**: CrossEntropyLoss for multi-class classification (pizza/steak/sushi)
    - **Device-agnostic**: Automatically detects CUDA/MPS/CPU

8) **Pretrained ViT Transfer Learning**
    - Use `torchvision.models.vit_b_16` with pretrained ImageNet weights
    - **Freeze base parameters** - only train the classification head
    - **Replace classifier head**: Linear(768, num_classes=3) for our food classes
    - **Pretrained transforms**: Use the same transforms the model was trained with
    - **Fine-tuning**: Much faster training (10 epochs) with lower learning rate (1e-3)
    - **Transfer learning advantage**: Leverages learned visual features from ImageNet

## Key Hyperparameters (ViT-Base)
- **Image Size**: 224×224
- **Patch Size**: 16×16 (196 patches total)
- **Embedding Dimension**: 768
- **Number of Heads**: 12
- **MLP Size**: 3072 (4x embedding dim)
- **Number of Layers**: 12
- **Dropout**: 0.1
