import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk

# Initial Parameters
threshold = 75
dilation_iteration = 2
kernel_size = 11
kernel2_size = 20

# Placeholder variables for the images
road_images = [None, None, None, None]  # Placeholder for 4 road images
back_img_gray = None  # Placeholder for the background image (gray)

# Function to load an image for a specific road or background
def load_image(road_index=None, is_background=False):
    global back_img_gray
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if filepath:
        if is_background:
            back_img_gray = cv2.imread(filepath, 0)  # Load as grayscale
        elif road_index is not None:
            road_images[road_index] = cv2.imread(filepath, 0)
        process_and_display_image()

# Function to process and display the image for each road
def process_and_display_image():
    global road_images, back_img_gray, threshold, dilation_iteration, kernel_size, kernel2_size

    if back_img_gray is None:
        print("Error: Background image is missing.")
        return

    vehicle_counts = []
    for idx, road_img in enumerate(road_images):
        if road_img is None:
            vehicle_counts.append(0)  # If no image for this road, set vehicle count to 0
            continue

        # Make a copy of the road image to draw bounding boxes
        road_img_copy = cv2.cvtColor(road_img, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for drawing colored boxes

        # Calculate absolute difference between the road image and the background
        image_diff = cv2.absdiff(road_img, back_img_gray)

        # Threshold the difference image to obtain a binary image
        _, binary_image = cv2.threshold(image_diff, threshold, 255, cv2.THRESH_BINARY)

        # Create kernels based on slider values
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        kernel2 = np.ones((kernel2_size, kernel2_size), np.uint8)

        # Apply morphological operations to clean up the image
        binary_af_morpho = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        binary_af_morpho = cv2.dilate(binary_af_morpho, kernel2, iterations=dilation_iteration)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_af_morpho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count the number of vehicles (contours) for the current road
        vehicle_counts.append(len(contours))

        # Draw bounding rectangles around the detected vehicles
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(road_img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangles

        # Resize the road image for display (to fit into the grid)
        road_img_copy_resized = cv2.resize(road_img_copy, (300, 200))  # Resize for fitting into GUI
        
        # Convert the OpenCV image to PIL format to display in the Tkinter window
        road_img_pil = Image.fromarray(cv2.cvtColor(road_img_copy_resized, cv2.COLOR_BGR2RGB))
        road_img_tk = ImageTk.PhotoImage(road_img_pil)

        # Update the label with the processed image
        road_labels[idx].config(image=road_img_tk)
        road_labels[idx].image = road_img_tk

    # Determine the road with the maximum vehicle count
    max_count = max(vehicle_counts)
    max_road_index = vehicle_counts.index(max_count) if max_count > 0 else -1

    # Adjust traffic light timers based on the road with the maximum vehicle count
    timer_adjustments = [0 if road_images[i] is None else 30 for i in range(4)]  # Set timer to 0 if no image
    if max_road_index != -1:
        timer_adjustments[max_road_index] += 10  # Add extra 10 seconds to the road with the highest count

    # Print results and timer adjustments
    print("===================================================================================")
    print("Vehicle counts per road:", vehicle_counts)
    print("Timer adjustments (seconds):", timer_adjustments)
    print("===================================================================================")

# Functions to update parameters from sliders
def update_threshold(val):
    global threshold
    threshold = int(val)
    process_and_display_image()

def update_dilation_iteration(val):
    global dilation_iteration
    dilation_iteration = int(val)
    process_and_display_image()

def update_kernel_size(val):
    global kernel_size
    kernel_size = int(val)
    process_and_display_image()

def update_kernel2_size(val):
    global kernel2_size
    kernel2_size = int(val)
    process_and_display_image()

# Setup Tkinter GUI
root = tk.Tk()
root.title("Smart Traffic Management System")
root.geometry("1200x600")  # Adjusted window size for left-right layout

# Create frames for left and right sections
left_frame = ttk.Frame(root)
left_frame.grid(row=0, column=0, padx=10, pady=5, sticky="n")

right_frame = ttk.Frame(root)
right_frame.grid(row=0, column=1, padx=10, pady=5, sticky="n")

# Labels to display the images for each road in a 2x2 grid (on the left side)
road_labels = [ttk.Label(left_frame) for _ in range(4)]
road_labels[0].grid(row=0, column=0, padx=5, pady=5)  # Top-left
road_labels[1].grid(row=0, column=1, padx=5, pady=5)  # Top-right
road_labels[2].grid(row=1, column=0, padx=5, pady=5)  # Bottom-left
road_labels[3].grid(row=1, column=1, padx=5, pady=5)  # Bottom-right

# Buttons to load images for each road (on the right side)
load_img_1_btn = ttk.Button(right_frame, text="Select Image for Road 1", command=lambda: load_image(0))
load_img_1_btn.grid(row=0, column=0, padx=10, pady=5)

load_img_2_btn = ttk.Button(right_frame, text="Select Image for Road 2", command=lambda: load_image(1))
load_img_2_btn.grid(row=1, column=0, padx=10, pady=5)

load_img_3_btn = ttk.Button(right_frame, text="Select Image for Road 3", command=lambda: load_image(2))
load_img_3_btn.grid(row=2, column=0, padx=10, pady=5)

load_img_4_btn = ttk.Button(right_frame, text="Select Image for Road 4", command=lambda: load_image(3))
load_img_4_btn.grid(row=3, column=0, padx=10, pady=5)

# Button to load the background image
load_back_img_btn = ttk.Button(right_frame, text="Select Background Image", command=lambda: load_image(is_background=True))
load_back_img_btn.grid(row=4, column=0, padx=10, pady=5)

# Sliders for threshold, dilation iteration, and kernel sizes
threshold_slider = tk.Scale(right_frame, from_=0, to=255, orient="horizontal", label="Threshold", command=update_threshold)
threshold_slider.set(threshold)
threshold_slider.grid(row=5, column=0, padx=10, pady=5)

dilation_slider = tk.Scale(right_frame, from_=1, to=10, orient="horizontal", label="Dilation Iterations", command=update_dilation_iteration)
dilation_slider.set(dilation_iteration)
dilation_slider.grid(row=6, column=0, padx=10, pady=5)

kernel_slider = tk.Scale(right_frame, from_=1, to=50, orient="horizontal", label="Kernel Size", command=update_kernel_size)
kernel_slider.set(kernel_size)
kernel_slider.grid(row=7, column=0, padx=10, pady=5)

kernel2_slider = tk.Scale(right_frame, from_=1, to=50, orient="horizontal", label="Kernel2 Size", command=update_kernel2_size)
kernel2_slider.set(kernel2_size)
kernel2_slider.grid(row=8, column=0, padx=10, pady=5)

root.mainloop()
