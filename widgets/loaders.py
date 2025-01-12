import json
import os
from fileinput import filename
from tkinter import Image

import ipywidgets as widgets
from ipyfilechooser import FileChooser
from ipywidgets import interact
from matplotlib import pyplot as plt

use_cpu_widget = widgets.Checkbox(
    value=True, description="Run model on CPU", disabled=False, indent=False
)

sigma_widget = widgets.IntSlider(
    value=25,
    min=0,
    max=75,
    step=1,
    description="std",
    disabled=False,
    continuous_update=False,
    orientation="horizontal",
    readout=True,
    readout_format="d",
)


def load_selected_path(state_file="filechooser_state.json"):
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            data = json.load(f)
            print(data.get("selected_path", None))
            return data.get("selected_path", None)
    return None


def get_file_chooser(state_file="filechooser_state.json"):
    # Function to save the selected path
    def save_selected_path(path):
        with open(state_file, "w") as f:
            json.dump({"selected_path": path}, f)


    # Callback function to handle file selection
    def on_file_selected(chooser):
        if chooser.selected:
            save_selected_path(chooser.value)
        else:
            print("No file selected")

    fc = FileChooser(path="./data")
    # Register the callback and display the widget
    fc.register_callback(on_file_selected)
    return fc

def browse_images(images, labels, rgb_den, figsize=(7, 7)):
    assert len(images) == len(labels), (
        "The number of images and labels must be the same"
    )
    n = len(images)

    def view_image(i):
        plt.figure(figsize=figsize)
        if rgb_den:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap="gray")
        plt.title(labels[i])
        plt.axis("off")
        plt.show()

    interact(view_image, i=(0, n - 1))