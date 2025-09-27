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
1) Data Loaders and Setup - Start out with image of (224, 224, 3) -> (height, width, colour_channels)

2) Patch and Positional Embedding
    - We want a smaller "patch" of image to be processed parallelly - we choose patch_size = 16
    - Expected output after patch embedding -> 2D tensor of (196, 768) -> ((14)^2, (16)^2 * 3) -> (number_of_patches, embedding_dimension = (patch_size)^2 * colour_channels)
    - Then run each patch through a Conv2D layer (kernel_size = patch_size)(stride = patch_size) -> creating a "feature map" of each patch = embedding layer 
    - output of this layer -> (batch_size, embedding_dimension, height_feature_map, width_feature_map) -> (1, 768, 14, 14)
    - Need to flatten (the spatial dimensions(14x14)) to a 1D sequence of flattened 2D patches - Linear layer of 196 patches (1, 768, 196) -> Patch embeddings!
    - Transpose Patch Embeddings to get sequence-first format - (1, 196. 768) 
    - These are trainable but we need a "classifier prediction" at the end so we also prepend a CLS token(basically) to the patch embedding layer - CLS: (1, 1, 768); patch_embeddings: (1, 196, 768)
    - Now, the Patch embedding + CLS layer looks like [CLS_token_embedding, patch_embeddings] = total size (number of patches + 1) (1, 197, 768)
    - Finally, adding a positional embedding to each patch (to know where it was in the image originally), create a tensor of the same shape as the Patch embedding + CLS layer (1, 197, 768) and just add to the patch embeddings layer
    
3) Multi-Head Attention (MSA)