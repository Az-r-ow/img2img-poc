# Img2Img POC

- [Img2Img POC](#img2img-poc)
  - [Quick Start](#quick-start)
  - [Examples](#examples)
    - [pix2pix](#pix2pix)
      - [Run](#run)
      - [Tips](#tips)
    - [Differential Diffusion](#differential-diffusion)
      - [Run](#run-1)
      - [Usage](#usage)
      - [Tips](#tips-1)

## Quick Start

Create an virtual env

```bash
python3 -m venv venv
```

or using `virtualenv`

```bash
virtualenv venv
```

Activate the venv

```bash
source ./venv/bin/activate
```

Install the requirements

```bash
python3 -m pip install -r requirements.txt
```

## Examples

### pix2pix

#### Run

Run the example with

```bash
python3 pix2pix_app.py
```

Then navigate to `<ip>:9000` and you should be able to see the UI.

This examples was duplicated from the following space ([click here](https://huggingface.co/spaces/timbrooks/instruct-pix2pix)).

It's using the `timbrooks/instruct-pix2pix` model, which is an instruction-based image editing model based on `stable_diffusion`.

**Github :** https://github.com/timothybrooks/instruct-pix2pix

#### Tips

If you're not getting the quality result you want, there may be a few reasons:

1. Is the image not changing enough? Your Image CFG weight may be too high. This value dictates how similar the output should be to the input. It's possible your edit requires larger changes from the original image, and your Image CFG weight isn't allowing that. Alternatively, your Text CFG weight may be too low. This value dictates how much to listen to the text instruction. The default Image CFG of 1.5 and Text CFG of 7.5 are a good starting point, but aren't necessarily optimal for each edit. Try:

- Decreasing the Image CFG weight, or
- Increasing the Text CFG weight, or

2. Conversely, is the image changing too much, such that the details in the original image aren't preserved? Try:

- Increasing the Image CFG weight, or
- Decreasing the Text CFG weight

3. Try generating results with different random seeds by setting "Randomize Seed" and running generation multiple times. You can also try setting "Randomize CFG" to sample new Text CFG and Image CFG values each time.
4. Rephrasing the instruction sometimes improves results (e.g., "turn him into a dog" vs. "make him a dog" vs. "as a dog").
5. Increasing the number of steps sometimes improves results.
6. Do faces look weird? The Stable Diffusion autoencoder has a hard time with faces that are small in the image. Try cropping the image so the face takes up a larger portion of the frame.

### Differential Diffusion

#### Run

```bash
python3 sd_diff_app.py
```

Then navigate to `<ip>:8000` for the ui

For this demo, I'm using the `stabilityai/stable-diffusion-2` model with the **Differential Diffusion** technique. Since it seems complex to implement I used the code provided by the researches team and hooked it to the demo.

#### Usage

Just upload the image and then the mapping will be generated automatically giving the most weights to the detected walls. The detection and map generation is done with a segmentation model called `facebook/maskformer-swin-large-ade`. And then both the initial image and the map will be passed to the Stable Diffusion 2 model.

#### Tips

The higher the "Guidance Scale" the more the image will respect the given text prompt.

However keeping a balanced value (between 7 and 12) can yield the best results. Sometimes going up till 15 is okay too !
