import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
from skimage import io
from skimage.color import rgb2gray
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure
import os
from main import convolve_sinogram, radon_transform, inverse_radon_transform,normalize_data,get_max
from scipy import ndimage

import pydicom
from pydicom.dataset import Dataset, FileDataset
import datetime


# Default parameter values
DEFAULT_STEP_ANGLE = 2.0  # Angular step of the emitter/detector system (in degrees)
DEFAULT_NUM_DETECTORS = 270  # Number of detectors (n)
DEFAULT_FAN_ANGLE = 135  # Spread of the emitter/detector system (l) (in degrees)
INTERMEDIATE_STEPS = 10  # Number of intermediate steps

class CTImageReconstructionApp:
    def __init__(self, root):


        self.patient_name = tk.StringVar()
        self.patient_id = tk.StringVar()
        self.exam_date = tk.StringVar(value=datetime.datetime.now().strftime('%Y%m%d'))
        self.comments = tk.StringVar()

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
        # DICOM metadata frame
        dicom_frame = ttk.LabelFrame(self.root, text="DICOM Metadata", padding=10)
        dicom_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Patient Name
        ttk.Label(dicom_frame, text="Patient Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.patient_name = tk.StringVar()
        name_entry = ttk.Entry(dicom_frame, textvariable=self.patient_name, width=20)
        name_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # Patient ID
        ttk.Label(dicom_frame, text="Patient ID:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.patient_id = tk.StringVar()
        id_entry = ttk.Entry(dicom_frame, textvariable=self.patient_id, width=15)
        id_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

        # Exam Date
        ttk.Label(dicom_frame, text="Exam Date (YYYYMMDD):").grid(row=0, column=4, sticky=tk.W, padx=5, pady=2)
        self.exam_date = tk.StringVar(value=datetime.datetime.now().strftime('%Y%m%d'))
        date_entry = ttk.Entry(dicom_frame, textvariable=self.exam_date, width=12)
        date_entry.grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)

        # Comments
        ttk.Label(dicom_frame, text="Comments:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.comments = tk.StringVar()
        comments_entry = ttk.Entry(dicom_frame, textvariable=self.comments, width=50)
        comments_entry.grid(row=1, column=1, columnspan=4, sticky=tk.W + tk.E, padx=5, pady=2)

        # Buttons for DICOM operations
        dicom_buttons_frame = ttk.Frame(dicom_frame)
        dicom_buttons_frame.grid(row=1, column=5, padx=5, pady=2)

        # Load DICOM Button
        load_dicom_btn = ttk.Button(dicom_buttons_frame, text="Load DICOM", command=self.load_dicom)
        load_dicom_btn.pack(side=tk.LEFT, padx=2)

        # Save DICOM Button
        save_dicom_btn = ttk.Button(dicom_buttons_frame, text="Save as DICOM", command=self.save_dicom)
        save_dicom_btn.pack(side=tk.LEFT, padx=2)

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
                self.filtered_sinogram = np.array(convolve_sinogram(normalize_data(self.sinogram.tolist(),get_max(self.sinogram))))
            else:
                self.filtered_sinogram = normalize_data(self.sinogram,get_max(self.sinogram))

            # Display sinograms
            self.ax2.clear()
            self.ax2.imshow(self.sinogram, cmap="gray", aspect='auto')
            self.ax2.set_title("Original Sinogram")
            self.ax2.axis('off')

            self.ax3.clear()
            self.ax3.imshow(self.filtered_sinogram, cmap="gray", aspect='auto')
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

    def load_dicom(self):
        """Load image from a DICOM file"""
        file_path = filedialog.askopenfilename(
            title="Choose DICOM file",
            filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.status_var.set(f"Loading DICOM image from {file_path}...")

                # Read the DICOM file
                ds = pydicom.dcmread(file_path)

                # Extract pixel data as image
                self.input_image = ds.pixel_array.astype(float)
                if len(self.input_image.shape) > 2:  # If it's an RGB image
                    self.input_image = rgb2gray(self.input_image)

                # Normalize to 0-1 range
                if self.input_image.max() > 0:
                    self.input_image = self.input_image / self.input_image.max()

                # Display original image
                self.ax1.clear()
                self.ax1.imshow(self.input_image, cmap='gray')
                self.ax1.set_title("Original Image (DICOM)")
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

                # Update DICOM metadata fields if available
                if hasattr(ds, 'PatientName'):
                    self.patient_name.set(str(ds.PatientName))
                if hasattr(ds, 'PatientID'):
                    self.patient_id.set(ds.PatientID)
                if hasattr(ds, 'StudyDate'):
                    self.exam_date.set(ds.StudyDate)
                if hasattr(ds, 'ImageComments'):
                    self.comments.set(ds.ImageComments)

                self.canvas.draw()
                self.status_var.set(f"DICOM image loaded successfully: {os.path.basename(file_path)}")
            except Exception as e:
                self.status_var.set(f"Error loading DICOM image: {str(e)}")

    def save_dicom(self):
        """Save the reconstructed image as a DICOM file with metadata"""
        if self.reconstructed_image is None:
            self.status_var.set("No reconstructed image to save.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save DICOM file",
            defaultextension=".dcm",
            filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.status_var.set(f"Saving reconstructed image as DICOM to {file_path}...")

                # Create file meta information
                file_meta = Dataset()
                file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
                file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

                # Create the FileDataset instance
                ds = FileDataset(file_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

                # Set patient info
                ds.PatientName = self.patient_name.get()
                ds.PatientID = self.patient_id.get()

                # Set study date
                ds.StudyDate = self.exam_date.get()

                # Set current time
                current_time = datetime.datetime.now().time()
                ds.StudyTime = current_time.strftime('%H%M%S.%f')

                # Set comments
                if self.comments.get():
                    ds.ImageComments = self.comments.get()

                # Set basic metadata needed for a valid DICOM file
                ds.Modality = 'CT'
                ds.SeriesInstanceUID = pydicom.uid.generate_uid()
                ds.StudyInstanceUID = pydicom.uid.generate_uid()
                ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

                # Set creation date/time
                dt = datetime.datetime.now()
                ds.ContentDate = dt.strftime('%Y%m%d')
                ds.ContentTime = dt.strftime('%H%M%S.%f')

                # Image-related attributes
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.PixelRepresentation = 0  # unsigned integer
                ds.HighBit = 7
                ds.BitsStored = 8
                ds.BitsAllocated = 8

                # Get the latest reconstructed image
                if self.intermediate_steps:
                    image_data = self.intermediate_steps[-1]
                else:
                    image_data = self.reconstructed_image

                # Gaussian blur before saving
                image_data = ndimage.gaussian_filter(image_data, sigma=0.5)

                # Scaling to 8 bits (0-255)
                image_data = (image_data * 255).astype(np.uint8)

                # Set pixel data
                ds.Rows, ds.Columns = image_data.shape
                ds.PixelData = image_data.tobytes()

                # Set transfer syntax
                ds.is_little_endian = True
                ds.is_implicit_VR = False

                # Save the DICOM file
                ds.save_as(file_path, write_like_original=False)

                self.status_var.set(f"DICOM file saved successfully: {os.path.basename(file_path)}")
            except Exception as e:
                self.status_var.set(f"Error saving DICOM file: {str(e)}")

