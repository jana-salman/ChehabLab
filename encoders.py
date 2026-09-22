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

    encoder.eval()

    return encoder, image_processor