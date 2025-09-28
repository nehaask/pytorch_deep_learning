"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import torch.nn as nn
from multiprocessing import freeze_support
import torch
import matplotlib.pyplot as plt
import data_setup, engine, model_builder, utils
from torchvision import transforms
from torchinfo import summary

# Setup hyperparameters
BATCH_SIZE = 32
IMG_SIZE = 224
PATCH_SIZE = 16

torch.manual_seed(42)

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

def main():
    # Setup device-agnostic code 
    if torch.cuda.is_available():
        device = "cuda" # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = "mps" # Apple GPU
    else:
        device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available
    print(device)

    # =========================================================================
    #### DATASET LOADING ####
    # =========================================================================

    # Create transform pipeline manually
    manual_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    print(f"Manually created transforms: {manual_transforms}")


    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=manual_transforms,
        batch_size=BATCH_SIZE
    )
    print(train_dataloader, test_dataloader, class_names)

    # Get a batch of images
    images_batch, labels_batch = next(iter(train_dataloader))
    image = images_batch[0]
    label = labels_batch[0]
    print(f"DataLoader: {image.shape}, {label}") 

    
    # Plot image with matplotlib
    plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
    plt.title(class_names[label])
    plt.axis(False)
    # plt.show()

    # =========================================================================
    ##### PATCH EMBEDDING #####
    # =========================================================================

    # Change image shape to be compatible with matplotlib (color_channels, height, width) -> (height, width, color_channels)
    image_permuted = image.permute(1, 2, 0)

    # Index to plot the top row of patched pixels
    plt.figure(figsize=(PATCH_SIZE, PATCH_SIZE))
    plt.imshow(image_permuted[:PATCH_SIZE, :, :]);
    # plt.show()

   # Setup hyperparameters and make sure IMG_SIZE and PATCH_SIZE are compatible
    num_patches = IMG_SIZE/PATCH_SIZE
    assert IMG_SIZE % PATCH_SIZE == 0, "Image size must be divisible by patch size"
    print(f"Number of patches per row: {num_patches}\
            \nNumber of patches per column: {num_patches}\
            \nTotal patches: {num_patches*num_patches}\
            \nPatch size: {PATCH_SIZE} pixels x {PATCH_SIZE} pixels")

    # Create a series of subplots
    fig, axs = plt.subplots(nrows=IMG_SIZE // PATCH_SIZE, # need int not float
                            ncols=IMG_SIZE // PATCH_SIZE,
                            figsize=(num_patches, num_patches),
                            sharex=True,
                            sharey=True)
            
    # Loop through height and width of image
    for i, patch_height in enumerate(range(0, IMG_SIZE, PATCH_SIZE)): # iterate through height
        for j, patch_width in enumerate(range(0, IMG_SIZE, PATCH_SIZE)): # iterate through width

            # Plot the permuted image patch (image_permuted -> (Height, Width, Color Channels))
            axs[i, j].imshow(image_permuted[patch_height:patch_height+PATCH_SIZE, # iterate through height
                                            patch_width:patch_width+PATCH_SIZE, # iterate through width
                                            :]) # get all color channels

            # Set up label information, remove the ticks for clarity and set labels to outside
            axs[i, j].set_ylabel(i+1,
                                rotation="horizontal",
                                horizontalalignment="right",
                                verticalalignment="center")
            axs[i, j].set_xlabel(j+1)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].label_outer()

    # Set a super title
    fig.suptitle(f"{class_names[label]} -> Patchified", fontsize=16)
    # plt.show()

    # Create an instance of patch embedding layer
    patchify = model_builder.PatchEmbedding(in_channels=3,
                            patch_size=16,
                            embedding_dim=768)

    # Pass a single image through
    print(f"Input image shape: {image.unsqueeze(0).shape}")
    patch_embedded_image = patchify(image.unsqueeze(0)) # add an extra batch dimension on the 0th index, otherwise will error
    print(f"Output patch embedding shape: {patch_embedded_image.shape}")

    utils.set_seeds()

    # # Create random input sizes
    # random_input_image = (1, 3, 224, 224)

    # # Get a summary of the input and outputs of PatchEmbedding
    # summary(model_builder.PatchEmbedding(),
    #         input_size=random_input_image,
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])

    # =========================================================================
    ##### ADDING A CLASS TOKEN EMBEDDING #####
    # =========================================================================
    # Get the batch size and embedding dimension
    batch_size = patch_embedded_image.shape[0]
    embedding_dimension = patch_embedded_image.shape[-1]

    # Create the class token embedding as a learnable parameter that shares the same size as the embedding dimension (D)
    class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension), # [batch_size, number_of_tokens, embedding_dimension]
                            requires_grad=True) # make sure the embedding is learnable

    # Print the class_token shape
    print(f"Class token shape: {class_token.shape} -> [batch_size, number_of_tokens, embedding_dimension]")

    # Add the class token embedding to the front of the patch embedding
    patch_embedded_image_with_class_embedding = torch.cat((class_token, patch_embedded_image),
                                                        dim=1) # concat on first dimension

    # print(f"Sequence of patch embeddings with class token prepended shape: {patch_embedded_image_with_class_embedding.shape} -> [batch_size, number_of_patches, embedding_dimension]")

    # =========================================================================
    ##### ADDING A POSITIONAL EMBEDDING #####
    # =========================================================================

    # Calculate N (number of patches)
    number_of_patches = int((IMG_SIZE* IMG_SIZE) / PATCH_SIZE**2)

    # Get embedding dimension
    embedding_dimension = patch_embedded_image_with_class_embedding.shape[2]

    # Create the learnable 1D position embedding
    position_embedding = nn.Parameter(torch.ones(1,
                                                number_of_patches+1,
                                                embedding_dimension),
                                    requires_grad=True) # make sure it's learnable
    
    # Add the position embedding to the patch and class token embedding
    patch_and_position_embedding = patch_embedded_image_with_class_embedding + position_embedding
    # print(patch_and_position_embedding)
    # print(f"Patch embeddings, class token prepended and positional embeddings added shape: {patch_and_position_embedding.shape} -> [batch_size, number_of_patches, embedding_dimension]")

    # =========================================================================
    ##### MULTI HEAD SELF ATTENTION (MSA) #####
    # =========================================================================

    # Create an instance of MSABlock
    multihead_self_attention_block = model_builder.MultiheadSelfAttentionBlock(embedding_dim=768,
                                                                num_heads=12)

    # Pass patch and position image embedding through MSABlock
    # patched_image_through_msa_block = multihead_self_attention_block(patch_and_position_embedding)
    # print(f"Input shape of MSA block: {patch_and_position_embedding.shape}")
    # print(f"Output shape MSA block: {patched_image_through_msa_block.shape}")

    # =========================================================================
    ##### MLP layer #####
    # =========================================================================

    # Create an instance of MLPBlock
    mlp_block = model_builder.MLPBlock(embedding_dim=768, 
                        mlp_size=3072, 
                        dropout=0.1)

    # Pass output of MSABlock through MLPBlock
    # patched_image_through_mlp_block = mlp_block(patched_image_through_msa_block)
    # print(f"Input shape of MLP block: {patched_image_through_msa_block.shape}")
    # print(f"Output shape MLP block: {patched_image_through_mlp_block.shape}")

    # =========================================================================
    ##### Transformer Block - combining the Multi-head Self attention layer + MLP layer #####
    # =========================================================================

    # Create an instance of TransformerEncoderBlock
    transformer_encoder_block = model_builder.TransformerEncoderBlock()

    # Print an input and output summary of our Transformer Encoder
    # summary(model=transformer_encoder_block,
    #         input_size=(1, 197, 768), # (batch_size, num_patches, embedding_dimension)
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])
    

    # =========================================================================
    ##### Using Torch's TransformerEncoder Layer architecture - for comparison
    # =========================================================================

    # Create the same as above with torch.nn.TransformerEncoderLayer()
    torch_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768, # Hidden size D from Table 1 for ViT-Base
                                                                nhead=12, # Heads from Table 1 for ViT-Base
                                                                dim_feedforward=3072, # MLP size from Table 1 for ViT-Base
                                                                dropout=0.1, # Amount of dropout for dense layers from Table 3 for ViT-Base
                                                                activation="gelu", # GELU non-linear activation
                                                                batch_first=True, # Batches come first
                                                                norm_first=True) # Normalize before MSA/MLP layers?

    # print(torch_transformer_encoder_layer)
    # Get the output of PyTorch's version of the Transformer Encoder
    # summary(model=torch_transformer_encoder_layer,
    #         input_size=(1, 197, 768), # (batch_size, num_patches, embedding_dimension)
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])
    

    # =========================================================================
    ##### ViT block - follow paper diagram for architecture details
    # =========================================================================

    utils.set_seeds()

    # Create a random tensor with same shape as a single image
    random_image_tensor = torch.randn(1, 3, 224, 224).to(device) # (batch_size, color_channels, height, width)

    # Create an instance of ViT with the number of classes (pizza, steak, sushi)
    vit = model_builder.ViT(num_classes=len(class_names)).to(device)
    vit(random_image_tensor)

    # Print a summary of our custom ViT model using torchinfo
    # summary(model=vit,
    #         input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
    #         # col_names=["input_size"],
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"]
    # )

    # =========================================================================
    ##### TRAINING - ViT 
    # =========================================================================

    # Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper
    optimizer = torch.optim.Adam(params=vit.parameters(),
                                lr=3e-3, # Base LR from Table 3 for ViT-* ImageNet-1k
                                betas=(0.9, 0.999), # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                                weight_decay=0.3) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k

    # Setup the loss function for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model and save the training results to a dictionary
    results = engine.train(model=vit,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=10,
                        device=device)
    
    # Plot our ViT model's loss curves
    utils.plot_loss_curves(results)

if __name__ == "__main__":
    freeze_support()   # only needed if making a frozen executable (e.g., PyInstaller)
    main()