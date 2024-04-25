import math
import random 
import torch
import gradio as gr
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline
from webcolors import CSS3_NAMES_TO_HEX  

model_id = "timbrooks/instruct-pix2pix"

help_text = """
Pick the color that you want to paint your walls in from the dropdown list. Once the color picked, you will see it in the "Chosen color" sample.
Generate the image with the `Generate Default` button. Doing so will generate the image with the following default parameters : 
- Seed : 1 
- Image CFG : 1
- Text CFG : 8
- Steps : 80

If the image doesn't change enough, regenerate with the `Change More` button. This will increment the variables that are responsible of image change by the following values :
- Text CFG : +1 **(Max: 12)**
- Steps : +5 **(Max: 100)**

Once the maximum values reached the image will not change any further.

If the image changes a lot, regenerate with the `Change Less` button. It will increment the following variable that's responsible of conserving the image properties :
- Image CFG : +0.2

Removing the image will reset the values to default.
"""

DEFAULT_INSTRUCTION = "paint the walls in : {color}"

DEFAULT_SEED = 1
DEFAULT_IMAGE_CFG = 1
DEFAULT_TEXT_CFG  = 8
DEFAULT_STEPS = 80

MAX_STEPS = 100
MAX_TEXT_CFG = 12

seed = DEFAULT_SEED
color = "black"

def main():
  pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
      model_id, torch_dtype=torch.float16, safety_checker=None
  ).to("cuda")

  dropdown_colors = CSS3_NAMES_TO_HEX.keys()
  color_hex = "#0000"

  def reset_hyperparameters(image_cfg, text_cfg, steps):
    return [DEFAULT_IMAGE_CFG, DEFAULT_TEXT_CFG, DEFAULT_STEPS]

  def generate(
      input_image: Image.Image,
      image_cfg,
      text_cfg,
      steps
  ):
    width, height = input_image.size
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(
        input_image, (width, height), method=Image.LANCZOS
    )

    instruction = DEFAULT_INSTRUCTION.replace("{color}", color)

    generator = torch.manual_seed(seed)
    edited_image = pipe(
      instruction,
      image=input_image,
      guidance_scale=text_cfg,
      image_guidance_scale=image_cfg,
      num_inference_steps=steps,
      generator=generator
    ).images[0]

    return edited_image.resize((width ,height))
  
  def generate_with_less_change(input_image,  image_cfg, text_cfg, steps):
    image_cfg += 0.2
    image_cfg = round(image_cfg, ndigits=2)
    edited_image = generate(input_image, image_cfg, text_cfg, steps)
    return [edited_image, image_cfg, text_cfg, steps]
  
  def generate_with_more_change(input_image, image_cfg, text_cfg, steps):
    text_cfg = text_cfg + 1 if text_cfg < MAX_TEXT_CFG else MAX_TEXT_CFG
    steps = steps + 5 if steps < MAX_STEPS else MAX_STEPS
    edited_image = generate(input_image, image_cfg, text_cfg, steps)
    return [edited_image, image_cfg, text_cfg, steps]
  
  def generate_default(input_image, image_cfg, text_cfg, steps):
    image_cfg, text_cfg, steps = reset_hyperparameters(image_cfg, text_cfg, steps)
    edited_image = generate(input_image,  image_cfg, text_cfg, steps)
    return [edited_image, image_cfg, text_cfg, steps]

  
  def disable_buttons(generate_more, generate_default_button,  generate_less):
    return [gr.update(interactive=False),gr.update(interactive=False), gr.update(interactive=False)]

  def enable_buttons(generate_more, generate_default_button, generate_less):
    return [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)]
  
  def select_color(selected_color: str):
    global color
    color = selected_color
    return CSS3_NAMES_TO_HEX[color]
  
  with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column(scale=1):
        color_dropdown = gr.Dropdown(dropdown_colors, label="Colors", info="Pick a color from the list")
      with gr.Column(scale=1):
        color_hex = gr.ColorPicker(label="Chosen color", interactive=False)
    with gr.Row():
      input_image = gr.Image(
          label="Input Image", type="pil", width=512, height=512, interactive=True
      )
      edited_image = gr.Image(
          label="Edited Image",
          type="pil",
          width=512,
          height=512,
          interactive=False,
      )
    with gr.Row():
      with gr.Column(scale=1):
        generate_less_button = gr.Button("Change Less", interactive=False)
      with gr.Column(scale=1):
        generate_default_button = gr.Button("Generate Default", interactive=False)
      with gr.Column(scale=1):
        generate_more_button = gr.Button("Change More", interactive=False)

    with gr.Row():
      with gr.Column(scale=1):
        image_cfg = gr.Number(value=DEFAULT_IMAGE_CFG, label="Image CFG", info="Increase with 'Generate Less'")
      with gr.Column(scale=1):
        text_cfg = gr.Number(value=DEFAULT_TEXT_CFG, label="Text CFG", info="Increase with 'Generate More'")
      with gr.Column(scale=1):
        steps = gr.Number(value=DEFAULT_STEPS, label="Steps", info="Increase with 'Generate More'")

    gr.Markdown(help_text)

    generate_default_button.click(
      fn=generate_default,
      inputs=[input_image, image_cfg, text_cfg, steps],
      outputs=[edited_image, image_cfg, text_cfg, steps]
    )

    generate_more_button.click(
      fn=generate_with_more_change,
      inputs=[input_image, image_cfg, text_cfg, steps],
      outputs=[edited_image, image_cfg, text_cfg, steps]
    )

    generate_less_button.click(
      fn=generate_with_less_change,
      inputs=[input_image, image_cfg, text_cfg, steps],
      outputs=[edited_image, image_cfg, text_cfg, steps]
    )

    color_dropdown.change(
      fn=select_color,
      inputs=[color_dropdown],
      outputs=[color_hex]
    )

    input_image.change(
      fn=reset_hyperparameters,
      inputs=[image_cfg, text_cfg, steps],
      outputs=[image_cfg, text_cfg, steps]
    )

    input_image.clear(
      fn=disable_buttons,
      inputs=[generate_more_button, generate_default_button, generate_less_button],
      outputs=[generate_more_button, generate_default_button, generate_less_button]
    )

    input_image.upload(
      fn=enable_buttons,
      inputs=[generate_more_button, generate_default_button, generate_less_button],
      outputs=[generate_more_button, generate_default_button, generate_less_button]
    )

    demo.queue()
    demo.launch(share=False, server_name="0.0.0.0", server_port=9000)


if __name__ == "__main__":
    main()
