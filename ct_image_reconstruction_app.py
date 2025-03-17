import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
from skimage import io
from skimage.color import rgb2gray
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure
import os
from main import convolve_sinogram, radon_transform, inverse_radon_transform, transpose_sinogram

# Default parameter values
DEFAULT_STEP_ANGLE = 2.0  # Angular step of the emitter/detector system (in degrees)
DEFAULT_NUM_DETECTORS = 270  # Number of detectors (n)
DEFAULT_FAN_ANGLE = 135  # Spread of the emitter/detector system (l) (in degrees)
INTERMEDIATE_STEPS = 10  # Number of intermediate steps


class CTImageReconstructionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CT Image Reconstruction")
        self.root.geometry("1200x800")

        self.input_image = None
        self.sinogram = None
        self.filtered_sinogram = None
        self.reconstructed_image = None
        self.intermediate_steps = []
        self.use_filtering = tk.BooleanVar(value=True)

        # Variables to store transformation parameters
        self.step_angle = tk.DoubleVar(value=DEFAULT_STEP_ANGLE)
        self.num_detectors = tk.IntVar(value=DEFAULT_NUM_DETECTORS)
        self.fan_angle = tk.DoubleVar(value=DEFAULT_FAN_ANGLE)

        self.create_widgets()

    def create_widgets(self):
        # Frame for control panel
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Button to load image
        load_btn = ttk.Button(control_frame, text="Load Image", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=5)

        # Checkbox for filtering
        filter_check = ttk.Checkbutton(control_frame, text="Use Filtering", variable=self.use_filtering)
        filter_check.pack(side=tk.LEFT, padx=5)

        # Adding fields to modify transformation parameters
        params_frame = ttk.LabelFrame(self.root, text="CT System Parameters", padding=10)
        params_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Emitter/detector system step
        ttk.Label(params_frame, text="System step Δα (°):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        step_entry = ttk.Entry(params_frame, textvariable=self.step_angle, width=8)
        step_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # Number of detectors
        ttk.Label(params_frame, text="Number of detectors (n):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        detectors_entry = ttk.Entry(params_frame, textvariable=self.num_detectors, width=8)
        detectors_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

        # System span
        ttk.Label(params_frame, text="System span (l) (°):").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        fan_entry = ttk.Entry(params_frame, textvariable=self.fan_angle, width=8)
        fan_entry.grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)

        # Button to process
        process_btn = ttk.Button(params_frame, text="Process Image", command=self.process_image)
        process_btn.grid(row=0, column=6, padx=20, pady=2)

        # Frame for step slider
        step_frame = ttk.Frame(self.root, padding=10)
        step_frame.pack(side=tk.TOP, fill=tk.X)

        # Step slider label
        ttk.Label(step_frame, text="Reconstruction steps:").pack(side=tk.LEFT, padx=5)

        # Step slider
        self.step_slider = ttk.Scale(step_frame, from_=0, to=INTERMEDIATE_STEPS - 1, orient=tk.HORIZONTAL,
                                     command=self.update_step_display, length=800)
        self.step_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.step_slider.set(0)

        # Step value label
        self.step_label = ttk.Label(step_frame, text="0/{}".format(INTERMEDIATE_STEPS - 1))
        self.step_label.pack(side=tk.LEFT, padx=5)

        # Image display frame
        display_frame = ttk.Frame(self.root, padding=10)
        display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Creating matplotlib figure to display images
        self.fig = Figure(figsize=(12, 8))

        # Space for original image
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax1.set_title("Original Image")
        self.ax1.axis('off')

        # Space for sinogram
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax2.set_title("Sinogram")
        self.ax2.axis('off')

        # Space for filtered sinogram
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax3.set_title("Filtered Sinogram")
        self.ax3.axis('off')

        # Space for reconstructed image
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        self.ax4.set_title("Reconstructed Image (Step 0)")
        self.ax4.axis('off')

        # Adding figure to window
        self.canvas = tkagg.FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Choose image file",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.status_var.set(f"Loading image from {file_path}...")
                self.input_image = io.imread(file_path)
                self.input_image = rgb2gray(self.input_image)

                # Display original image
                self.ax1.clear()
                self.ax1.imshow(self.input_image, cmap='gray')
                self.ax1.set_title("Original Image")
                self.ax1.axis('off')

                # Clear other plots
                self.ax2.clear()
                self.ax2.set_title("Sinogram")
                self.ax2.axis('off')

                self.ax3.clear()
                self.ax3.set_title("Filtered Sinogram")
                self.ax3.axis('off')

                self.ax4.clear()
                self.ax4.set_title("Reconstructed Image")
                self.ax4.axis('off')

                self.canvas.draw()
                self.status_var.set(f"Image loaded successfully: {os.path.basename(file_path)}")
            except Exception as e:
                self.status_var.set(f"Error loading image: {str(e)}")

    def process_image(self):
        if self.input_image is None:
            self.status_var.set("Please load an image first.")
            return

        try:
            # Get current parameter values from GUI
            step_angle = self.step_angle.get()
            num_detectors = self.num_detectors.get()
            fan_angle = self.fan_angle.get()

            # Calculate number of steps based on angular step (for full 360° rotation)
            num_steps = int(180 / step_angle)

            self.status_var.set(
                f"Calculating sinogram: Δα={step_angle}°, n={num_detectors}, l={fan_angle}°, {num_steps} steps...")
            self.root.update()

            # Calculate sinogram with given parameters
            self.sinogram = radon_transform(
                self.input_image,
                num_steps=num_steps,
                span=fan_angle,
                num_rays=num_detectors
            )

            # Apply filtering if enabled
            if self.use_filtering.get():
                self.status_var.set("Filtering sinogram...")
                self.root.update()
                self.filtered_sinogram = np.array(convolve_sinogram(self.sinogram.tolist()))
            else:
                self.filtered_sinogram = self.sinogram

            # Display sinograms
            self.ax2.clear()
            self.ax2.imshow(transpose_sinogram(self.sinogram), cmap="gray", aspect='auto')
            self.ax2.set_title("Original Sinogram")
            self.ax2.axis('off')

            self.ax3.clear()
            self.ax3.imshow(transpose_sinogram(self.filtered_sinogram), cmap="gray", aspect='auto')
            self.ax3.set_title("Filtered Sinogram" if self.use_filtering.get() else "Original Sinogram")
            self.ax3.axis('off')

            # Image reconstruction and saving intermediate steps
            self.status_var.set("Reconstructing image with intermediate steps...")
            self.root.update()

            self.reconstructed_image, self.intermediate_steps = inverse_radon_transform(
                self.filtered_sinogram, self.input_image,
                num_steps=num_steps, span=fan_angle, num_rays=num_detectors,
                save_intermediate=True
            )

            # Ensure exactly INTERMEDIATE_STEPS steps
            if len(self.intermediate_steps) > INTERMEDIATE_STEPS:
                indices = np.linspace(0, len(self.intermediate_steps) - 1, INTERMEDIATE_STEPS, dtype=int)
                self.intermediate_steps = [self.intermediate_steps[i] for i in indices]
            elif len(self.intermediate_steps) < INTERMEDIATE_STEPS:
                # Duplicate last step if we have fewer steps
                last_step = self.intermediate_steps[-1]
                while len(self.intermediate_steps) < INTERMEDIATE_STEPS:
                    self.intermediate_steps.append(last_step)

            # Update slider range and display first step
            self.step_slider.configure(to=len(self.intermediate_steps) - 1)
            self.step_slider.set(0)
            self.update_step_display(0)

            self.status_var.set("Image processing completed.")
        except Exception as e:
            self.status_var.set(f"Error during processing: {str(e)}")

    def update_step_display(self, value):
        if not self.intermediate_steps:
            return

        step = int(float(value))
        self.step_label.config(text=f"{step}/{len(self.intermediate_steps) - 1}")

        # Display intermediate reconstruction step
        self.ax4.clear()
        self.ax4.imshow(self.intermediate_steps[step], cmap='gray')
        self.ax4.set_title(f"Reconstructed Image (Step {step + 1})")
        self.ax4.axis('off')

        self.canvas.draw()
