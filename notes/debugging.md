# PyTorch Debugging Mental Notes 🐛🔧

## 1. **Tensor Shape Debugging**
```python
# Always check shapes first!
print(f"Input shape: {x.shape}")
print(f"Expected: {expected_shape}")

# Quick shape inspection
x.shape  # torch.Size([batch, channels, height, width])
x.size() # same as .shape
x.dim()  # number of dimensions
x.numel() # total number of elements
```

**Common Shape Issues:**
- Missing batch dimension: `x.unsqueeze(0)` 
- Wrong dimension order: `x.permute(1, 2, 0)` for matplotlib
- Dimension mismatch in matmul: check inner dimensions match
- Conv2D expects: `(batch, channels, height, width)`

## 2. **Device Debugging**

**The command depends on whether you're checking a Tensor or a nn.Module:**

🔹 **For a Tensor:**
```python
print(tensor.device)
```
Example:
```python
x = torch.randn(3, 3).to("mps")
print(x.device)   # mps:0
```

🔹 **For a nn.Module (like a model):**
PyTorch doesn't expose a direct `.device` for models since different submodules can be on different devices. The common trick is to check the first parameter (or buffer):
```python
next(model.parameters()).device
```
Example:
```python
print(next(pretrained_vit.parameters()).device)  # mps:0
```

**Complete Device Debugging:**
```python
# Check device location
print(f"Model device: {next(model.parameters()).device}")
print(f"Data device: {x.device}")

# Move to same device
x = x.to(device)
model = model.to(device)

# Debug mixed device errors
model.device  # doesn't exist! Use next(model.parameters()).device
```

**Device Mental Checklist:**
- Model and data on same device?
- Loss function inputs on same device?
- Check after every `.to(device)` call
- CPU tensors can't use CUDA operations

## 3. **Gradient Debugging**
```python
# Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
    else:
        print(f"{name}: NO GRADIENT!")

# Enable/disable gradients
with torch.no_grad():  # disable gradients
    predictions = model(x)

x.requires_grad_(True)  # enable gradients for tensor
```

**Gradient Issues:**
- `loss.backward()` called?
- `optimizer.zero_grad()` before each batch?
- Check for `detach()` calls breaking gradient flow
- Frozen parameters: `param.requires_grad = False`

## 4. **DataLoader Debugging**
```python
# Inspect a single batch
batch = next(iter(dataloader))
print(f"Batch type: {type(batch)}")
print(f"Batch length: {len(batch)}")

# Check data shapes and types
images, labels = batch
print(f"Images: {images.shape}, dtype: {images.dtype}")
print(f"Labels: {labels.shape}, dtype: {labels.dtype}")

# Visualize first image
plt.imshow(images[0].permute(1, 2, 0))  # CHW -> HWC for matplotlib
```

**DataLoader Checklist:**
- Batch size makes sense?
- Images normalized properly? (0-1 or -1 to 1)
- Labels in correct format? (class indices, not one-hot)
- Transform pipeline working?

## 5. **Loss Function Debugging**
```python
# Check loss value and gradients
loss = criterion(outputs, targets)
print(f"Loss: {loss.item()}")
print(f"Loss requires_grad: {loss.requires_grad}")

# NaN/Inf debugging
torch.isnan(loss).any()  # Check for NaN
torch.isinf(loss).any()  # Check for infinity

# Check output ranges
print(f"Output range: {outputs.min().item()} to {outputs.max().item()}")
```

**Loss Issues:**
- NaN losses → check learning rate, data normalization
- Loss not decreasing → learning rate too low/high
- CrossEntropyLoss expects raw logits (no softmax)
- MSELoss expects same shape for prediction and target

## 6. **Model Architecture Debugging**
```python
# Use torchinfo for detailed summary
from torchinfo import summary
summary(model, input_size=(batch_size, channels, height, width))

# Forward pass debugging
def debug_forward(model, x):
    for name, layer in model.named_children():
        x = layer(x)
        print(f"{name}: {x.shape}")
    return x

# Check parameter counts
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params}, Trainable: {trainable_params}")
```

## 7. **Training Loop Debugging**
```python
# Essential training loop checks
for epoch in range(epochs):
    model.train()  # Set to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Clear gradients
        
        # Debug data
        print(f"Batch {batch_idx}: data {data.shape}, target {target.shape}")
        
        output = model(data)
        loss = criterion(output, target)
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Invalid loss at batch {batch_idx}")
            break
            
        loss.backward()
        optimizer.step()
```

**Training Checklist:**
- `model.train()` before training, `model.eval()` before validation
- `optimizer.zero_grad()` before `loss.backward()`
- No `torch.no_grad()` during training (blocks gradients)
- Learning rate scheduled properly?

## 8. **Memory Debugging**
```python
# Check GPU memory usage
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated()}")
    print(f"GPU cached: {torch.cuda.memory_reserved()}")
    torch.cuda.empty_cache()  # Free unused memory

# Memory-efficient validation
with torch.no_grad():  # Saves memory
    model.eval()
    for batch in val_loader:
        predictions = model(batch)
```

**Memory Issues:**
- CUDA out of memory → reduce batch size
- Memory leak → check for unnecessary `.retain_grad()`
- Large models → use gradient checkpointing
- Validation → always use `torch.no_grad()`

## 9. **Common Error Patterns**

**RuntimeError: Expected all tensors to be on the same device**
```python
# Fix: Move everything to same device
x = x.to(device)
model = model.to(device)
```

**RuntimeError: mat1 and mat2 shapes cannot be multiplied**
```python
# Debug matrix multiplication
print(f"Matrix 1: {A.shape}")
print(f"Matrix 2: {B.shape}")
# Fix: A.shape[-1] must equal B.shape[-2]
```

**IndexError: Target is out of bounds**
```python
# Check label range
print(f"Labels min: {labels.min()}, max: {labels.max()}")
print(f"Num classes: {num_classes}")
# Fix: labels should be 0 to num_classes-1
```

## 10. **Quick Debug Functions**
```python
def tensor_info(tensor, name="Tensor"):
    """Quick tensor debugging info"""
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, requires_grad={tensor.requires_grad}")
    print(f"  Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    print(f"  Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")

def model_summary(model):
    """Quick model info"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable: {trainable:,}")
    print(f"Device: {next(model.parameters()).device}")

def check_gradients(model):
    """Check if gradients are flowing"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"{name}: {grad_norm:.6f}")
            else:
                print(f"{name}: NO GRADIENT!")
```

## 11. **Mental Debugging Workflow**
1. **Check shapes** → most common issue
2. **Check devices** → model and data on same device?
3. **Check data** → visualize a batch, check ranges/types
4. **Check gradients** → flowing properly?
5. **Check loss** → reasonable values? NaN/Inf?
6. **Check memory** → GPU memory usage?
7. **Isolate problem** → test components individually

## 12. **Prevention Tips**
- **Use assertions**: `assert x.shape == expected_shape`
- **Add debug prints**: Remove after fixing
- **Use `torchinfo.summary()`**: Catch architecture issues early
- **Test with small batch**: Debug with batch_size=1 first
- **Reproducible bugs**: Set `torch.manual_seed(42)`
- **Save intermediate outputs**: Debug layer by layer