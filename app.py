import streamlit as st

import tempfile
import torch
import numpy as np
from imageio import imread
from PIL import Image
import os

from models import Generator
from utils.util import InputData, saveOutput

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache
def getModel(channelExponent=5, trained_model=None):

    print("Loading the trained Model")
    netG = Generator(channelExponent=int(channelExponent)).to(device)

    if trained_model is None:
        trained_model_path = f"trained_models/modelG_{channelExponent}.pth"
    # else:
    #     print("else trained_model ")
    #     trained_model_path=f"trained_models/{trained_model}"

    netG.load_state_dict(torch.load(trained_model_path, map_location=device))
    netG.eval()
    return netG


def predictFlow(np_arr, model):
    input_data = InputData(np_arr, removePOffset=True, makeDimLess=True)
    input_arr, target_arr = input_data.input, input_data.target
    input_tensor = torch.tensor(input_arr)
    input_tensor = input_tensor.to(device)
    input_batch = input_tensor.unsqueeze(0)

    print("Running Inference Model")
    output_batch = model(input_batch.float())

    output_tr = output_batch.squeeze(0)
    output_tr = output_tr.to(device)
    output_arr = output_tr.detach().cpu().numpy()

    print("Saving output")
    saveOutput(output_arr, target_arr)
    return 1


def solver(np_arr, trained_model=None, channelExponent=5):
    model = getModel(channelExponent, trained_model)
    result_id = predictFlow(np_arr, model)
    return result_id


result_path = os.path.join(os.getcwd(), "results")
if not (os.path.exists(result_path)):
    os.makedirs(result_path)

st.sidebar.title("Deep Flow Prediction Application")

trained_model = None
channelExponent = st.sidebar.selectbox("Select channelExponent", ("5", "7"))
# st.sidebar.write("OR")
# trained_model_list = os.listdir("./trained_models/")
# trained_model = st.sidebar.selectbox("Model",trained_model_list)

ux = st.sidebar.number_input("Enter Ux")
uy = st.sidebar.number_input("Enter Uy")

airfoil_files = os.listdir("./example/airfoils")[:100]
airfoil_files = [x.replace(".png", "") for x in airfoil_files]

airfoil_type = st.sidebar.selectbox("Airfoil type", airfoil_files)

st.sidebar.write("OR")
airfoil_img_file = st.sidebar.file_uploader("Choose airfoil file (Binary Image)", type="png")

my_expander = st.sidebar.expander("Upload input file in npz format")
upload_file = my_expander.file_uploader("Choose file...")

st.sidebar.write("")
sts = st.sidebar.button("Submit")

if sts:
    if upload_file:
        file = upload_file.read()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file)
        np_arr = np.load(tfile.name)["a"]
        resid = solver(np_arr, channelExponent=channelExponent)
        print(resid)
        if resid == 1:
            st.image("results/result.png")

    elif ux and uy and (airfoil_type or airfoil_img_file):
        if airfoil_type:
            im = imread(f"example/airfoils/{airfoil_type}.png")
        else:
            image = Image.open(airfoil_img_file)
            im = np.array(image)

        np_im = np.array(im)
        np_im = np_im / np.max(np_im)

        c1, c2 = st.columns(2)
        c1.header("Airfoil Geometry")
        print(f"Input Image shape: {np_im.shape}")
        c1.image(np_im, use_column_width=True)

        np_im = np.flipud(np_im).transpose()  # in model's input format

        ux = float(ux)
        uy = float(uy)

        fx = np.full((128, 128), ux) * np_im
        fy = np.full((128, 128), uy) * np_im

        np_im = 1 - np_im

        np_arr = np.stack((fx, fy, np_im))

        result_id = solver(np_arr, trained_model, channelExponent)
        print(result_id)

        if result_id == 1:
            print("displaying results")
            col1, col2, col3 = st.columns(3)
            col1.header("Velocity X")
            col1.image("results/result_velX_pred.png", use_column_width=True)
            col2.header("Velocity Y")
            col2.image("results/result_velY_pred.png", use_column_width=True)
            col3.header("Pressure")
            col3.image("results/result_pressure_pred.png", use_column_width=True)
