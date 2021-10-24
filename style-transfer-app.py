from __future__ import print_function

import torchvision.models as models
import matplotlib.pyplot as plt
import argparse
import streamlit as st

import config
from model import *
from utils import *

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("--contentImage", type=str, required=True, help="path to content image")
# ap.add_argument("--styleImage", type=str, required=True, help="path to style image")
# ap.add_argument("--deep", type=int, default=300, help="the deep level of generated image related to content and style images")
# ap.add_argument("--styleWeight", type=int, default=1, help="style weight")
# ap.add_argument("--contentWeight", type=int, default=1, help="content weight")
# args = vars(ap.parse_args())

# content_img = content_image_loader(args["contentImage"])
# style_img = style_image_loader(args["styleImage"], args["contentImage"])
# input_img = content_img.clone()

# # define CNN model
# cnn = models.vgg19(pretrained=True).features.to(config.DEVICE).eval()

# output = run_style_transfer(cnn, config.CNN_NORMALIZATION_MEAN, config.CNN_NORMALIZATION_STD,
#                             content_img, style_img, input_img,
#                             style_weight=args["styleWeight"]*1e6, content_weight=args["contentWeight"], num_steps=args["deep"])

# plt.figure()
# imshow(output, title='Output Image')
# plt.ioff()
# plt.show()

st.title("Style transfer application")
st.markdown("This application allows you to generate images based on content and style images you've given.")

# upload and show content and style images
uploaded_content_image = st.file_uploader(label="Please upload an content image", type="jpg")
if uploaded_content_image is not None:
    content_image = Image.open(uploaded_content_image, 'r')
    st.image(content_image, use_column_width=True, channels='RGB')
    content_img = content_image_loader(uploaded_content_image)
    input_img = content_img.clone()

uploaded_style_image = st.file_uploader(label="Please upload an style image", type="jpg")
if uploaded_style_image is not None:
    style_image = Image.open(uploaded_style_image, 'r')
    st.image(style_image, use_column_width=True, channels='RGB')
    style_img = style_image_loader(uploaded_style_image, uploaded_content_image)

# user input features
deep = st.sidebar.selectbox("Deep level of generated image based on content and style images",
                        [300, 500, 700, 1000])
style_weight = st.sidebar.slider("Style weight", 1.0, 5.0, 1.0)
content_weight = st.sidebar.slider("Content weight", 1.0, 5.0, 1.0)

# define CNN model
cnn = models.vgg19(pretrained=True).features.to(config.DEVICE).eval()

# run the model
if st.button("Start generating"):
    with st.spinner("[INFO] Starting style transfer computation..."):
        output = run_style_transfer(cnn, config.CNN_NORMALIZATION_MEAN, config.CNN_NORMALIZATION_STD,
                                content_img, style_img, input_img,
                                style_weight=style_weight*1e6, content_weight=content_weight, num_steps=deep)
        output = output.cpu().clone()
        output = output.squeeze(0)
        output = unloader(output)
        st.balloons()
    st.subheader("Generated image")
    st.image(output, use_column_width=True, channels='RGB')