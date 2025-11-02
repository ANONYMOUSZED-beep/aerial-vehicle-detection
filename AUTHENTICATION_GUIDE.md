# Hugging Face Authentication Setup

The rebotnix/rb_vehicle model is a gated repository that requires authentication to access.

## Steps to Access the Gated Model:

### 1. Create Hugging Face Account
- Go to https://huggingface.co/join
- Create an account if you don't have one

### 2. Request Access to the Model
- Visit: https://huggingface.co/rebotnix/rb_vehicle
- Click "Request access" button
- Fill out the access request form
- Wait for approval from the model authors

### 3. Generate Access Token
- Go to https://huggingface.co/settings/tokens
- Create a new token with "Read" permissions
- Copy the token (starts with hf_...)

### 4. Login via CLI (Recommended)
```bash
# In your activated virtual environment
huggingface-cli login
# Enter your token when prompted
```

### 5. Alternative: Set Environment Variable
```bash
# Windows Command Prompt
set HF_TOKEN=hf_your_token_here

# Windows PowerShell  
$env:HF_TOKEN="hf_your_token_here"

# Or add to your Python code:
# os.environ["HF_TOKEN"] = "hf_your_token_here"
```

### 6. Verify Access
```python
from huggingface_hub import whoami
print(whoami())
```

## Alternative Models (If Access Denied)

If access to rebotnix/rb_vehicle is denied, here are alternative aerial vehicle detection models:

1. **facebook/detr-resnet-50** - General object detection DETR model
2. **microsoft/table-transformer-detection** - Can be fine-tuned for vehicles  
3. **hustvl/yolos-tiny** - YOLO-like transformer model
4. **nvidia/mit-b0** - Segmentation model that can detect vehicles

## Notes
- The gated model likely has better performance for aerial imagery
- Request access ASAP as approval can take time
- Keep your token secure and don't commit it to version control