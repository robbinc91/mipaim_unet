import sys
import os
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import nibabel as nib
from nilearn import masking, image
from nilearn.plotting import view_img, glass_brain, plot_anat, plot_epi, find_cut_slices, find_xyz_cut_coords, cm
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
from utils.losses import *
from utils.preprocess import *
from keras.models import load_model
import tempfile

matplotlib.use('TkAgg')


class App:

    def __run_predict(self, filepath):
        if not self.model_path:
            return None
        model = load_model(self.model_path,
                           custom_objects={'dice_coefficient': dice_coefficient, 'dice_loss': dice_loss})

        _input = to_uint8(get_data(filepath))[None, None, ...]
        _output = model.predict(_input)
        return _output.squeeze()

    def __init__(self):
        self.model_path = None
        self.app_title = 'Custom MRI processor'
        self.x_index, self.y_index, self.z_index = 10, 10, 10
        self.actual_axis = 2  # z
        self.loaded_image = False

        self.im = None
        self.canvas = None
        self.img = None
        self.loaded_image = False

        self.im = None
        self.canvas = None
        self.z_index = 0

        self.window = tk.Tk()

        self.window.title(f"{self.app_title}")

        self.menubar = tk.Menu(self.window)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Open", command=self.open_file)
        self.filemenu.add_command(label="Save")
        self.filemenu.add_command(label="Exit")

        self.maskmenu = tk.Menu(self.menubar, tearoff=0)
        self.maskmenu.add_command(label="Add mask", command=self.open_mask_file)

        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.menubar.add_cascade(label="Mask", menu=self.maskmenu)

        self.window.config(menu=self.menubar)

        self.window.rowconfigure(0, minsize=600, weight=1)
        self.window.columnconfigure(1, minsize=600, weight=1)

        self.window.minsize(width=800, height=600)
        self.central_frame_container = tk.Frame(self.window)
        self.central_frame = tk.Frame(self.central_frame_container)

        self.left_buttons = tk.Frame(self.window, relief=tk.RIDGE, borderwidth=2)
        self.btn_open = tk.Button(self.left_buttons, text="Open", command=self.open_file)
        self.btn_open_mask = tk.Button(self.left_buttons, text="Open mask", command=self.open_mask_file)
        self.btn_process_file = tk.Button(self.left_buttons, text="Process", command=self.process_file)
        self.btn_load_model = tk.Button(self.left_buttons, text="Open Model", command=self.__load_model)
        self.btn_save = tk.Button(self.left_buttons, text="Save As...", command=self.save_file)

        self.btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.btn_open_mask.grid(row=1, column=0, sticky="ew", padx=5)
        self.btn_process_file.grid(row=2, column=0, sticky="ew", padx=5)
        self.btn_load_model.grid(row=3, column=0, sticky="ew", padx=5)
        self.btn_save.grid(row=4, column=0, sticky="ew", padx=5)

        self.right_buttons = tk.Frame(self.window, relief=tk.RIDGE, borderwidth=2)

        self.right_buttons.columnconfigure(0, minsize=200)
        self.left_buttons.columnconfigure(0, minsize=200)

        self.btn_exit = tk.Button(self.right_buttons, text="Exit", command=self.quit)

        self.btn_exit.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.left_buttons.grid(row=0, column=0, sticky="ns")
        self.central_frame_container.grid(row=0, column=1, sticky="nwse")
        self.central_frame.grid(row=0, column=0, sticky="nwse")
        self.right_buttons.grid(row=0, column=2, sticky="ns")

        self.central_frame_container.rowconfigure(0, weight=1)
        self.central_frame_container.columnconfigure(0, weight=1)

        # central_frame_container.rowconfigure(1, weight=1)

        self.central_frame.rowconfigure(0, weight=1)
        self.central_frame.columnconfigure(0, weight=1)

        self.central_frame.rowconfigure(1, weight=1)
        self.central_frame.columnconfigure(1, weight=1)

        self.frame = tk.Frame(self.central_frame, relief=tk.RIDGE, borderwidth=2)
        self.frame2 = tk.Frame(self.central_frame, relief=tk.RIDGE, borderwidth=2)
        self.frame3 = tk.Frame(self.central_frame, relief=tk.RIDGE, borderwidth=2)
        # bottom_frame = tk.Frame(central_frame_container)

        # bottom_frame.grid(row=1, column= 0, sticky="nswe")
        self.frame.grid(row=0, column=0, sticky="nwes")
        self.frame2.grid(row=0, column=1, sticky="nwes")
        self.frame3.grid(row=1, column=0, sticky="nwse")

        # frame.bind("<MouseWheel>", on_mousewheel,  add='+')
        self.frame.bind("<Enter>", self.register_axial_mousewheel)
        self.frame.bind("<Leave>", self.unregister_all_mousewheel)

        self.frame2.bind("<Enter>", self.register_coronal_mousewheel)
        self.frame2.bind("<Leave>", self.unregister_all_mousewheel)

        self.frame3.bind("<Enter>", self.register_sagital_mousewheel)
        self.frame3.bind("<Leave>", self.unregister_all_mousewheel)

    def save_file(self):
        """Save the current file as a new file."""
        filepath = asksaveasfilename(
            defaultextension="txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )
        if not filepath:
            return
        with open(filepath, "w") as output_file:
            text = self.txt_edit.get("1.0", tk.END)
            output_file.write(text)
        self.window.title(f"{self.app_title} - {filepath}")

    def __apply_mask_from_file(self, filepath):
        self.img_shape = self.img.shape

        self.mask = nib.load(filepath)
        self.mask_shape = self.mask.shape
        if self.mask_shape != self.img_shape:
            # Display error msg
            print('not the same shape')
            return False

        fig = plt.figure(1, facecolor='black', edgecolor='white')
        fig.clear()

        self.img.get_fdata()[self.mask.get_fdata() > 0.5] = 1000

        self.im = plt.imshow(self.img.get_fdata()[:, :, self.z_index], cmap='gray')

        self.__update_canvas()
        return True

    def __update_canvas(self):
        self.canvas.draw()
        self.canvas2.draw()
        self.canvas3.draw()

    def open_mask_file(self):
        if not self.loaded_image:
            return False
        filepath = askopenfilename(
            filetypes=[("Nifti Files", "*.nii", "*.gz")]
        )
        if not filepath:
            return False

        return self.__apply_mask_from_file(filepath)

    def open_file(self):
        self.window.resizable(width="false", height="false")
        """Open a file for editing."""
        filepath = askopenfilename(
            filetypes=[("Nifti Files", "*.nii", '*.gz')]
        )

        if not filepath:
            return False

        self.image_path = filepath
        self.__open_file(filepath)
        return True

    def __open_file(self, filepath):
        self.loaded_image = True
        self.actual_axis = 2
        self.img = nib.load(filepath)

        self.x_index, self.y_index, self.z_index = self.img.shape
        self.x_index //= 2
        self.y_index //= 2
        self.z_index //= 2

        val = self.im

        fig = plt.figure(1, figsize=(10, 10), dpi=100, facecolor='black', edgecolor='white')
        fig.clear()
        self.im = plt.imshow(self.img.get_fdata()[:, :, self.z_index], cmap='gray', aspect='equal', resample=False,
                             origin="lower")

        fig2 = plt.figure(100, figsize=(10, 10), dpi=100, facecolor='black', edgecolor='white')
        fig2.clear()
        self.im2 = plt.imshow(self.img.get_fdata()[:, self.y_index, :], cmap='gray', aspect='equal', resample=False)

        fig3 = plt.figure(1000, figsize=(10, 10), dpi=100, facecolor='black', edgecolor='white')
        fig3.clear()
        self.im3 = plt.imshow(self.img.get_fdata()[self.x_index, :, :], cmap='gray', aspect='equal', resample=False)

        if val is not None:
            pass
            # im.set_data(img.get_fdata()[:,:,z_index])
            # fig = plt.figure(1, figsize=(10, 10), dpi=100, facecolor='black', edgecolor='white')
        else:
            self.canvas = FigureCanvasTkAgg(fig, master=self.frame)  # A tk.DrawingArea.
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            self.canvas2 = FigureCanvasTkAgg(fig2, master=self.frame2)  # A tk.DrawingArea.
            self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            self.canvas3 = FigureCanvasTkAgg(fig3, master=self.frame3)  # A tk.DrawingArea.
            self.canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.__update_canvas()

        self.window.title(f"{self.app_title} - {filepath.split('/')[-1]}")
        self.window.resizable(width="true", height="true")

    def quit(self, *args):

        self.window.quit()
        self.window.destroy()
        self.im = None
        self.canvas = None

    def __load_model(self):
        filepath = askopenfilename(
            filetypes=[("h5 Files", "*.h5")]
        )

        if not filepath:
            return False
        self.model_path = filepath
        return True

    def process_file(self):
        if not self.loaded_image:
            return False

        if not self.model_path:
            self.__load_model()

        dta = self.__run_predict(self.image_path)
        dta_tmp = nib.Nifti1Image(dta, self.img.affine)
        nib.save(dta_tmp, os.sep.join([tempfile.gettempdir(), 'temp_pred_.nii.gz']))
        self.__apply_mask_from_file(os.sep.join([tempfile.gettempdir(), 'temp_pred_.nii.gz']))



    def on_mousewheel(self, event):
        if not self.loaded_image:
            return

        delta = int(event.delta)
        self.actual_frame = self.active_frame

        if self.actual_frame == 'axial':
            # im = globals()['im']

            if delta > 0:
                self.z_index += 1
            else:
                self.z_index -= 1
            self.z_index %= self.img.shape[2]
            self.im.set_data(self.img.get_fdata()[:, :, self.z_index])

            self.canvas.draw()

        elif self.actual_frame == 'coronal':
            # im2 = globals()['im2']
            if delta > 0:
                self.y_index += 1
            else:
                self.y_index -= 1

            self.y_index %= self.img.shape[1]
            self.im2.set_data(self.img.get_fdata()[:, self.y_index, :])

            # canvas2 = globals()['canvas2']
            self.canvas2.draw()
            # globals()['y_index'] = y_index

        else:

            # im3 = globals()['im3']
            if delta > 0:
                self.x_index += 1
            else:
                self.x_index -= 1

            self.x_index %= self.img.shape[0]
            # print('x_index:', x_index)
            self.im3.set_data(self.img.get_fdata()[self.x_index, :, :])
            # canvas3 = globals()['canvas3']
            self.canvas3.draw()
            # globals()['x_index'] = x_index

    def set_coronal_view(self, event=None):
        if not self.loaded_image:
            return

        self.actual_axis = 1

        if self.im:
            self.im.set_data(self.img.get_fdata()[:, self.y_index, :])
            self.canvas.draw()

    def set_sagital_view(self, event=None):
        if not self.loaded_image:
            return

        if self.im:
            self.im.set_data(self.img.get_fdata()[self.x_index, :, :])
            self.canvas.draw()

    def set_axial_view(self, event=None):
        if not self.loaded_image:
            return

        if self.im:
            self.im.set_data(self.img.get_fdata()[:, :, self.z_index])
            self.canvas.draw()

    def mouse_enter(self, event):
        print(event)

    def register_axial(self, event=None):
        # print('register_axial')
        self.frame_axial_view.bind_all('<Button-1>', self.set_axial_view, add='+')

    def unregister(self, event=None):
        print('unregister_axial')
        self.frame_axial_view.unbind_all('<Button-1>')

    def register_coronal(self, event=None):
        self.frame_axial_view.bind_all('<Button-1>', self.set_coronal_view, add='+')

    def register_sagital(self, event=None):
        self.frame_axial_view.bind_all('<Button-1>', self.set_sagital_view, add='+')

    def register_axial_mousewheel(self, event=None):
        print('register axial mousewheel')
        self.central_frame.bind_all("<MouseWheel>", self.on_mousewheel, add='+')
        self.active_frame = 'axial'

    def register_coronal_mousewheel(self, event=None):
        self.central_frame.bind_all("<MouseWheel>", self.on_mousewheel, add='+')
        self.active_frame = 'coronal'

    def register_sagital_mousewheel(self, event=None):
        self.central_frame.bind_all("<MouseWheel>", self.on_mousewheel, add='+')
        self.active_frame = 'sagital'

    def unregister_all_mousewheel(self, event=None):
        self.central_frame.unbind_all("<MouseWheel>")

    def run(self):
        self.window.mainloop()


app = App()
app.window.mainloop()