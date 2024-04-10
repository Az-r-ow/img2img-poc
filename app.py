import math
import random 
import torch 
import gradio as gr
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline
from webcolors import hex_to_name

model_id = "timbrooks/instruct-pix2pix"


def main():

  pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")
  example_image = Image.open("imgs/office.jpg").convert("RGB")

  def load_example(
    steps: int,
    randomize_seed: bool,
    seed: int,
    randomize_cfg: bool,
    text_cfg_scale: float,
    image_cfg_scale: float,
):
    example_instruction = "Pain the wall in blue"
    return [example_image, example_instruction] + generate(
        example_image,
        example_instruction,
        steps,
        randomize_seed,
        seed,
        randomize_cfg,
        text_cfg_scale,
        image_cfg_scale,
    )

  def generate(
      input_image: Image.Image,
      instruction: str,
      color: str, 
      steps: int,
      randomize_seed: bool,
      seed: int,
      randomize_cfg: bool,
      text_cfg_scale: float,
      image_cfg_scale: float
  ):
    seed = random.randint(0, 100000) if randomize_seed else seed
    text_cfg_scale = round(random.uniform(6.0, 9.0), ndigits=2) if randomize_cfg else text_cfg_scale
    image_cfg_scale = round(random.uniform(1.2, 1.8), ndigits=2) if randomize_cfg else image_cfg_scale

    width, height = input_image.size
    # width, height = (100, 100)
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    if instruction == "":
        return [input_image, seed]
    
    color_placeholder = "{color}"

    if color_placeholder in instruction:
      try:
        color = hex_to_name(color)
      except ValueError as e:
          print(e)
      instruction = instruction.replace(color_placeholder, color)

    generator = torch.manual_seed(seed)
    edited_image = pipe(
        instruction, image=input_image,
        guidance_scale=text_cfg_scale, image_guidance_scale=image_cfg_scale,
        num_inference_steps=steps, generator=generator,
    ).images[0]

    return [seed, text_cfg_scale, image_cfg_scale, edited_image]
  
  def reset():
     return [0, "Randomize Seed", 1242, "Fix CFG", 7.5, 1.5, None]
  
  with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column(scale=1, min_width=100):
          generate_button = gr.Button("Generate")
      with gr.Column(scale=1, min_width=100):
          load_button = gr.Button("Load Example")
      with gr.Column(scale=1, min_width=100):
          reset_button = gr.Button("Reset")
      with gr.Column(scale=2):
          instruction = gr.Textbox(lines=1, label="Edit Instruction", interactive=True)
      with gr.Column(scale=1):
          color = gr.ColorPicker(label="Pick a color", interactive=True)

    with gr.Row():
      input_image = gr.Image(label="Input Image", type="pil", width=512, height=512, interactive=True)
      edited_image = gr.Image(label="Edited Image", type="pil",width=512, height=512, interactive=False)

    with gr.Row():
      steps = gr.Number(value=50, precision=0, label="Steps", interactive=True)
      randomize_seed = gr.Radio(
          ["Fix Seed", "Randomize Seed"],
          value="Randomize Seed",
          type="index",
          show_label=False,
          interactive=True,
      )
      seed = gr.Number(value=1371, precision=0, label="Seed", interactive=True)
      randomize_cfg = gr.Radio(
          ["Fix CFG", "Randomize CFG"],
          value="Fix CFG",
          type="index",
          show_label=False,
          interactive=True,
      )
      text_cfg_scale = gr.Number(value=7.5, label="Text CFG", interactive=True)
      image_cfg_scale = gr.Number(value=1.5, label="Image CFG", interactive=True)

    load_button.click(
        fn=load_example,
        inputs=[
            steps,
            randomize_seed,
            seed,
            randomize_cfg,
            text_cfg_scale,
            image_cfg_scale,
        ],
        outputs=[input_image, instruction, seed, text_cfg_scale, image_cfg_scale, edited_image],
    )
    
    generate_button.click(
        fn=generate,
        inputs=[
            input_image,
            instruction,
            color,
            steps,
            randomize_seed,
            seed,
            randomize_cfg,
            text_cfg_scale,
            image_cfg_scale,
        ],
        outputs=[seed, text_cfg_scale, image_cfg_scale, edited_image],
    )

    reset_button.click(
        fn=reset, 
        inputs=[],
        outputs=[steps, randomize_seed, seed, randomize_cfg, text_cfg_scale, image_cfg_scale, edited_image],
    )

    demo.queue()
    demo.launch(share=False, server_name="0.0.0.0", server_port=9000)


if __name__ == "__main__":
    main()
