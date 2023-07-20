'''
*****************************************************************************
GENERATING IMAGES FROM A SUMMARIZED TEXT EXTRACTED FROM A USER UPLOADED PDF
*****************************************************************************

This program performs the following tasks:

1. Creates a Web user interface
2. Upload a PDF from the Web interface
3. Extract the text from all pages of the PDF
4. Concatenate the text of all pages in a single document
5. Pass the extracted text to a Large Langauge Model to summarize it
6. Pass the summarized text to a generative AI model to create an image representing the summarized text

Last updated: 2023-07-14
'''

from typing import Optional
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from sd2.generate_original import PIPELINE_NAMES, generate
import openai
import fitz         # this is pymupdf
from dotenv import load_dotenv, find_dotenv
import os
import sys
import dotenv



openai.api_key = os.getenv("OPENAI_API_KEY")
_ = load_dotenv(find_dotenv())
# OPENAI_API_KEY = ''
# openai.api_key = OPENAI_API_KEY
print(openai.api_key)


DEFAULT_PROMPT = "The main characters in the story are Ginger, a kind and smart giraffe with a long neck and long legs, and Toby, a tired and hungry monkey."
DEFAULT_WIDTH, DEFAULT_HEIGHT = 512, 512
OUTPUT_IMAGE_KEY = "output_img"
LOADED_IMAGE_KEY = "loaded_image"


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def get_prompt(text):
    prompt_03 = f"""
            Your task is to output a single sentence describing the two main characters in the story and their environment
            below, delimited by triple
            backticks.
            Do it in at most 35 words in a wording suitable for a generative art algorithm like stable diffusion.
            Story: ```{text}```
            Main characters description:
            """
    response = get_completion(prompt_03)
    print(response)
    return response


def get_image(key: str) -> Optional[Image.Image]:
    if key in st.session_state:
        return st.session_state[key]
    return None


def set_image(key: str, img: Image.Image):
    st.session_state[key] = img


def upload_and_generate_button(prefix, pipeline_name: PIPELINE_NAMES, **kwargs):
    # prompt_value =""
    uploaded_pdf = st.file_uploader("Load pdf: ", type=['pdf'])
    if uploaded_pdf is not None:
        doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        prompt = get_prompt(text)
        negative_prompt = ""
        st.write(prompt)
    else:
        prompt = DEFAULT_PROMPT
        negative_prompt = ""

    # prompt = st.text_area(
    #     "Prompt",
    #     value= DEFAULT_PROMPT,  # change to if null DEFAULT, else  prompt_value   *************************************
    #     key=f"{prefix}-prompt",
    # )
    # negative_prompt = st.text_area(
    #     "Negative prompt",
    #     value="",
    #     key=f"{prefix}-negative-prompt",
    # )
    col1, col2 = st.columns(2)
    with col1:
        steps = st.slider("Number of inference steps", min_value=1, max_value=200, value=50, key=f"{prefix}-inference-steps")
    with col2:
        guidance_scale = st.slider(
            "Guidance scale", min_value=0.0, max_value=20.0, value=7.5, step=0.5, key=f"{prefix}-guidance-scale"
        )
    enable_attention_slicing = st.checkbox('Enable attention slicing (enables higher resolutions but is slower)',
                                           key=f"{prefix}-attention-slicing",
                                           value=True)
    enable_xformers = st.checkbox('Enable xformers library (better memory usage)',
                                  key=f"{prefix}-xformers",
                                  value=False)

    if st.button("Generate image", key=f"{prefix}-btn"):
        with st.spinner("Generating image..."):
            image = generate(
                prompt,
                pipeline_name,
                negative_prompt=negative_prompt,
                steps=steps,
                guidance_scale=guidance_scale,
                enable_attention_slicing=enable_attention_slicing,
                enable_xformers=enable_xformers,
                **kwargs,
            )
            set_image(OUTPUT_IMAGE_KEY, image.copy())
        st.image(image)


def width_and_height_sliders(prefix):
    col1, col2 = st.columns(2)
    with col1:
        width = st.slider(
            "Width",
            min_value=64,
            max_value=1600,
            step=16,
            value=512,
            key=f"{prefix}-width",
        )
    with col2:
        height = st.slider(
            "Height",
            min_value=64,
            max_value=1600,
            step=16,
            value=512,
            key=f"{prefix}-height",
        )
    return width, height


def image_uploader(prefix):
    image = st.file_uploader("Image", ["jpg", "png"], key=f"{prefix}-uploader")
    if image:
        image = Image.open(image)
        print(f"loaded input image of size ({image.width}, {image.height})")
        image = image.resize((DEFAULT_WIDTH, DEFAULT_HEIGHT))
        return image

    return get_image(LOADED_IMAGE_KEY)


def txt2img_tab():
    prefix = "txt2img"
    width, height = width_and_height_sliders(prefix)
    upload_and_generate_button(prefix, "txt2img", width=width, height=height)


def main():
    st.set_page_config(layout="wide")
    st.title("Image Generation -- Gliese.AI")
    # st.write(openai.api_key)
    txt2img_tab()

    with st.sidebar:
        st.header("Latest Output Image")
        output_image = get_image(OUTPUT_IMAGE_KEY)
        if output_image:
            st.image(output_image)
            if st.button("Use this image for inpainting and img2img"):
                set_image(LOADED_IMAGE_KEY, output_image.copy())
                st.experimental_rerun()
        else:
            st.markdown("No output generated yet")


if __name__ == "__main__":
    main()
