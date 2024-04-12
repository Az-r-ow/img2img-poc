import gradio as gr
import torch
from torchvision import transforms
from SD2.diff_pipe import StableDiffusionDiffImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler
from PIL import Image, ImageFilter, ImageOps
from transformers import pipeline

NUM_INFERENCE_STEPS = 50
device = "cuda"

segment = pipeline("image-segmentation", model="facebook/maskformer-swin-large-ade")

base = StableDiffusionDiffImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to(device)

base.scheduler = DPMSolverMultistepScheduler.from_config(base.scheduler.config)


def segment_image_generate_mapping(img):
    predictions = segment(img)
    greyscale_image = None
    for i in range(len(predictions)):
        if pred[i]["label"] == "wall":
            inverted_image = ImageOps.invert(pred[i]["mask"])
            greyscale_image = inverted_image.convert("LA").convert("RGB")
            # Apply a Gaussian blur to the binary mask to smooth the boundaries
            greyscale_image = greyscale_image.point(
                lambda p: p - 10 if p > 127.5 else p + 75
            )
            smoothed_mask = greyscale_image.filter(
                ImageFilter.BoxBlur(5)
            )  # Adjust radius as needed

            # Paste the smoothed mask onto a black background to create the smoothed image
            smoothed_image = Image.new("L", greyscale_image.size, color=0)
            smoothed_image.paste(smoothed_mask, (0, 0))
            greyscale_image = smoothed_image
    return greyscale_image


def preprocess_image(image):
    image = image.convert("RGB")
    image = transforms.CenterCrop((image.size[1] // 64 * 64, image.size[0] // 64 * 64))(
        image
    )
    image = transforms.ToTensor()(image)
    image = image * 2 - 1
    image = image.unsqueeze(0).to(device)
    return image


def preprocess_map(map):
    map = map.convert("L")
    map = transforms.CenterCrop((map.size[1] // 64 * 64, map.size[0] // 64 * 64))(map)
    # convert to tensor
    map = transforms.ToTensor()(map)
    map = map.to(device)
    return map


def inference(image, gs, prompt, negative_prompt):
    map = segment_image_generate_mapping(image)
    validate_inputs(image, map)
    image = preprocess_image(image)
    map = preprocess_map(map)
    edited_images = base(
        prompt=prompt,
        image=image,
        strength=1,
        guidance_scale=gs,
        num_images_per_prompt=1,
        negative_prompt=negative_prompt,
        map=map,
        num_inference_steps=NUM_INFERENCE_STEPS,
        output_type="latent",
    ).images[0]
    return map, edited_images


def validate_inputs(image, map):
    if image is None:
        raise gr.Error("Missing image")
    if map is None:
        raise gr.Error("Missing map")


example1 = [
    "assets/input2.jpg",
    "assets/map2.jpg",
    17.5,
    "Tree of life under the sea, ethereal, glittering, lens flares, cinematic lighting, artwork by Anna Dittmann & Carne Griffiths, 8k, unreal engine 5, hightly detailed, intricate detailed",
    "bad anatomy, poorly drawn face, out of frame, gibberish, lowres, duplicate, morbid, darkness, maniacal, creepy, fused, blurry background, crosseyed, extra limbs, mutilated, dehydrated, surprised, poor quality, uneven, off-centered, bird illustration, painting, cartoons",
]
example2 = [
    "assets/input3.jpg",
    "assets/map4.png",
    21,
    "overgrown atrium, nature, ancient black marble columns and terracotta tile floors, waterfall, ultra-high quality, octane render, corona render, UHD, 64k",
    "Two bodies, Two heads, doll, extra nipples, bad anatomy, blurry, fuzzy, extra arms, extra fingers, poorly drawn hands, disfigured, tiling, deformed, mutated, out of frame, cloned face, watermark, text, lowres, disfigured, ostentatious, ugly, oversaturated, grain, low resolution, blurry, bad anatomy, poorly drawn face, mutant, mutated,  blurred, out of focus, long neck, long body, ugly, disgusting, bad drawing, childish",
]
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(label="Input Image", type="pil")
                change_map = gr.Image(label="Change Map", type="pil", interactive=False)
            gs = gr.Slider(0, 28, value=7.5, label="Guidance Scale")
            prompt = gr.Textbox(label="Prompt")
            neg_prompt = gr.Textbox(label="Negative Prompt")
            with gr.Row():
                clr_btn = gr.ClearButton(
                    components=[input_image, change_map, gs, prompt, neg_prompt]
                )
                run_btn = gr.Button("Run", variant="primary")

        output = gr.Image(label="Output Image")
    gr.Markdown("Differential Diffusion with Stable Diffusion 2.")
    run_btn.click(
        inference,
        inputs=[input_image, gs, prompt, neg_prompt],
        outputs=[change_map, output],
    )

    clr_btn.add(output)
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=8000)
