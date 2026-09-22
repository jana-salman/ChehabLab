from transformers import AutoImageProcessor, AutoModel
import open_clip
import torch


def get_encoder(encoder_id, device=None):

    # Choose GPU if available, otherwise CPU
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # BioMedCLIP uses OpenCLIP instead of Hugging Face Transformers
    if "biomedclip" in encoder_id.lower():

        encoder, _, image_processor = open_clip.create_model_and_transforms(
            f"hf-hub:{encoder_id}"
        )

        encoder = encoder.to(device)

    # Google ViT, OpenAI CLIP, DINOv3 and RAD-DINO
    else:

        image_processor = AutoImageProcessor.from_pretrained(
            encoder_id
        )

        encoder = AutoModel.from_pretrained(
            encoder_id
        ).to(device)

    encoder.eval() #switch to evaluation mode: dropout and batchnorm are altered accordingly,
                   #but does not stop tracking gradients.

    return encoder, image_processor

def get_features(encoder, X, layer):

    #X must be: (batch_size, channels, height, width) the images
    if len(X.shape) != 4:
        raise Exception("The function expects a tensor of 4 dimensions.")

    #Put the images on the same device as the model
    device = next(encoder.parameters()).device
    X = X.to(device)

    features = []

    # Save the output of the selected layer
    def hook(module, input, output):
        features.append(output)

    # Attach the hook
    handle = layer.register_forward_hook(hook)

    try:
        with torch.no_grad():

            # OpenAI CLIP
            if hasattr(encoder, "vision_model"):
                encoder.vision_model(pixel_values=X)

            # BioMedCLIP
            elif hasattr(encoder, "encode_image"):
                encoder.encode_image(X)

            # Google ViT, DINOv3, RAD-DINO
            else:
                encoder(pixel_values=X)

    finally:
        # Always remove the hook
        handle.remove()

    output = features[0]

    # Some transformer layers return a tuple
    if isinstance(output, tuple):
        output = output[0]

    # For transformer output: (B, tokens, hidden_size)
    # take token 0 = CLS token
    if len(output.shape) == 3:
        output = output[:, 0, :]

    return output

def test_(encoder_id):
    #Load the model and its preprocessor
    encoder, image_processor = get_encoder(encoder_id)

    #Create a dummy batch of 2 images: (B, 3, H, W)
    size = 518 if "rad-dino" in encoder_id.lower() else 224
    X = torch.rand(2, 3, size, size)

    #Choose the last transformer block layer
    #OpenAI CLIP
    if hasattr(encoder, "vision_model"):        
        layer = encoder.vision_model.encoder.layers[-1]
    #BioMedCLIP
    elif hasattr(encoder, "visual"):            
        layer = encoder.visual.trunk.blocks[-1]
    #Google ViT, RAD-DINO
    elif hasattr(encoder, "encoder"):           
        layer = encoder.encoder.layer[-1]
    #DINOv3
    else:                                       
        layer = encoder.layer[-1]

    #Extract features:
    features = get_features(encoder, X, layer)

    print(f"{encoder_id}: input {tuple(X.shape)} -> features {tuple(features.shape)}")
    return features