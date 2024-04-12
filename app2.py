import gradio as gr
import torch
from torchvision import transforms
from SD2.diff_pipe import StableDiffusionDiffImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler

NUM_INFERENCE_STEPS = 50
device = "cuda"

base = StableDiffusionDiffImg2ImgPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to(device)


base.scheduler = DPMSolverMultistepScheduler.from_config(base.scheduler.config)
# refiner.scheduler = DPMSolverMultistepScheduler.from_config(base.scheduler.config)


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


def inference(image, map, gs, prompt, negative_prompt):
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
    # edited_images = refiner(
    #     prompt=prompt,
    #     original_image=image,
    #     image=edited_images,
    #     strength=1,
    #     guidance_scale=7.5,
    #     num_images_per_prompt=1,
    #     negative_prompt=negative_prompt,
    #     map=map,
    #     num_inference_steps=NUM_INFERENCE_STEPS,
    #     denoising_start=0.8,
    # ).images[0]
    return edited_images


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
                change_map = gr.Image(label="Change Map", type="pil")
            gs = gr.Slider(0, 28, value=7.5, label="Guidance Scale")
            prompt = gr.Textbox(label="Prompt")
            neg_prompt = gr.Textbox(label="Negative Prompt")
            with gr.Row():
                clr_btn = gr.ClearButton(
                    components=[input_image, change_map, gs, prompt, neg_prompt]
                )
                run_btn = gr.Button("Run", variant="primary")

        output = gr.Image(label="Output Image")
    gr.Examples(
        examples=[example1, example2],
        inputs=[input_image, change_map, gs, prompt, neg_prompt],
    )
    gr.Markdown(
        "Differential Diffusion with SDXL; Thanks to the community for the prompts in the examples."
    )
    run_btn.click(
        inference,
        inputs=[input_image, change_map, gs, prompt, neg_prompt],
        outputs=output,
    )
    clr_btn.add(output)
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=9000)
