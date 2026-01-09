# Construct GUI executable 

import customtkinter
from tkinterdnd2 import *
from CTkMenuBar import *

import asyncio
import threading
import traceback
import os
import json
import sys
import numpy as np
from typing import Callable, Union
import process_audio
from misc.log import *

from multiprocessing import freeze_support

class Spinbox(customtkinter.CTkFrame):
    def __init__(self, *args,
                 width: int = 100,
                 height: int = 32,
                 min: float = -np.Inf,
                 max: float = np.Inf,
                 type: f'int',
                 step_size: Union[int, float] = 1,
                 command: Callable = None,
                 **kwargs):
        super().__init__(*args, width=width, height=height, **kwargs)

        self.step_size = step_size
        self.command = command
        self.min = min
        self.max = max
        self.type = type

        self.configure(fg_color=("gray78", "gray28"))
        self.grid_columnconfigure((0, 2), weight=0) 
        self.grid_columnconfigure(1, weight=1)

        self.subtract_button = customtkinter.CTkButton(self, text="-", width=height-6, height=height-6,
                                                       command=self.subtract_button_callback)
        self.subtract_button.grid(row=0, column=0, padx=(3, 0), pady=3)

        self.entry = customtkinter.CTkEntry(self, justify=customtkinter.CENTER, width=width-(2*height), height=height-6, border_width=0)
        self.entry.grid(row=0, column=1, columnspan=1, padx=3, pady=3, sticky="ew")

        self.add_button = customtkinter.CTkButton(self, text="+", width=height-6, height=height-6,
                                                  command=self.add_button_callback)
        self.add_button.grid(row=0, column=2, padx=(0, 3), pady=3)

        if self.type == 'int':
            self.entry.insert(0, "0")
        elif self.type == ' float':
            self.entry.insert(0, "0.0")

    def add_button_callback(self):
        if self.command is not None:
            self.command()
        try:
            if self.type == 'int':
                value = (int(self.entry.get()) + self.step_size)
                if value <= self.max:
                    self.entry.delete(0, "end")
                    self.entry.insert(0, str(value))
            elif self.type == 'float':
                value = (float(self.entry.get()) + self.step_size)
                if value <= self.max:
                    self.entry.delete(0, "end")
                    self.entry.insert(0, float_to_str(value, float(self.step_size)))
        except ValueError:
            return

    def subtract_button_callback(self):
        if self.command is not None:
            self.command()
        try:
            if self.type == 'int':
                value = (int(self.entry.get()) - self.step_size)
                if value >= self.min:
                    self.entry.delete(0, "end")
                    self.entry.insert(0, str(value))
            elif self.type == 'float':
                value = (float(self.entry.get()) - self.step_size)
                if value >= self.min:
                    self.entry.delete(0, "end")
                    self.entry.insert(0, float_to_str(value, float(self.step_size)))
        except ValueError:
            return

    def get(self) -> Union[float, None]:
        try:
            if self.type == 'int':
                return int(self.entry.get())
            elif self.type == 'float':
                return float(self.entry.get())
        except ValueError:
            return None

    def set(self, value: float):
        self.entry.delete(0, "end")
        if self.type == 'int':
            self.entry.insert(0, str(int(value)))
        elif self.type == 'float':
            self.entry.insert(0, float_to_str(float(value), float(self.step_size)))
    
def float_to_str(value: float, step_size: float):
    value = float(value)
    digits = len(str(step_size).split('.')[-1])
    return(f"{round(value, digits):.{digits}f}")

class ConsoleRedirector:
    def __init__(self, textbox):
        self.textbox = textbox
        self.textbox.tag_config('error', background="red", foreground="black")
        self.textbox.tag_config('success', background="green", foreground="black")
        self.textbox.tag_config('warning', background="yellow", foreground="black")
    
    def write(self, message):
        self.textbox.configure(state="normal")
        message = message.replace("[0m", "")
        if "[31m" in message:
            message = message.replace("[31m", "")
            self.textbox.insert(customtkinter.END, message, 'error')
        elif "[32m" in message:
            message = message.replace("[32m", "")
            self.textbox.insert(customtkinter.END, message, 'success')
        elif "[33m" in message:
            message = message.replace("[33m", "")
            self.textbox.insert(customtkinter.END, message, 'warning')
        else:
            self.textbox.insert(customtkinter.END, message)
        self.textbox.yview(customtkinter.END)  # scroll to bottom
        self.textbox.configure(state="disabled")
        
    def flush(self):
        pass # Handle flush calls from the print function

class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        # Variables
        self.extract_segments = customtkinter.IntVar(self, 1)
        self.retain_logit_score = customtkinter.IntVar(self, 1)
        self.use_target_model = customtkinter.IntVar(self, 1)
        self.use_ensemble = customtkinter.IntVar(self, 1)

        self.title("Bioacoustic Model Ensemble Interface")
        self.geometry("1180x600")

        menu = CTkMenuBar(master=self, bg_color='#222222')
        menu_button_file = menu.add_cascade("File")
        menu_button_about = menu.add_cascade("About", postcommand=self.callback_about_popup)
        menu_dropdown_file = CustomDropdownMenu(widget=menu_button_file)
        menu_dropdown_file.add_option(option="Open session...", command=self.open_session_file)
        menu_dropdown_file.add_option(option="Save session", command=self.save_session_file)

        # Frames
        self.frame_top = customtkinter.CTkFrame(self)
        self.frame_bottom = customtkinter.CTkFrame(self)
        self.frame_top.pack(side='top', fill = 'x', expand = False, padx=0, pady=0)
        self.frame_bottom.pack(side='top', fill = 'both', expand = True, padx=0, pady=0)

        self.frame_io = customtkinter.CTkFrame(self.frame_top, fg_color='transparent')
        self.frame_config = customtkinter.CTkFrame(self.frame_top)

        self.frame_io.pack(side='left', fill = 'both', expand = False, padx=5, pady=5)
        self.frame_config.pack(side='left', fill = 'both', expand = True, padx=5, pady=5)

        self.console_frame = customtkinter.CTkFrame(self.frame_bottom)
        self.console_frame.pack(side='top', fill = 'both', expand = True, padx=5, pady=5)
        self.console_textbox = customtkinter.CTkTextbox(self.console_frame, fg_color="#111111")
        self.console_textbox.configure(state="disabled")
        self.console_textbox.pack(side='top', fill = 'both', expand = True, padx=5, pady=5)
        sys.stdout = ConsoleRedirector(self.console_textbox)
        sys.stderr = ConsoleRedirector(self.console_textbox)
        self.progressbar = customtkinter.CTkProgressBar(master=self.console_frame)
        self.progressbar.pack(side='top', fill = 'x', expand = False, padx=5, pady=(0,5))
        self.progressbar.configure(mode="determinate")
        self.progressbar.set(0.0)

        self.frame_input = customtkinter.CTkFrame(self.frame_io)
        self.frame_output = customtkinter.CTkFrame(self.frame_io)
        self.frame_process = customtkinter.CTkFrame(self.frame_io)
        self.frame_input.pack(side='top', fill = 'both', expand = True, padx=0, pady=(0,5))
        self.frame_output.pack(side='top', fill = 'both', expand = True, padx=0, pady=(5,5))
        self.frame_process.pack(side='top', fill = 'both', expand = True, padx=0, pady=(5,0))

        self.tabview_model = customtkinter.CTkTabview(self.frame_config)
        self.frame_options = customtkinter.CTkFrame(self.frame_config)
        self.frame_model_config = customtkinter.CTkFrame(self.frame_config)
        self.frame_model_config.pack(side='left', fill = 'both', expand = True, padx=(10,5), pady=5)
        self.frame_options.pack(side='left', fill = 'both', expand = False, padx=(5,10), pady=5)

        # Input
        self.label_input = customtkinter.CTkLabel(self.frame_input, text="Input audio data")
        self.label_input.pack(side='top')
        self.entry_in_path = customtkinter.CTkEntry(self.frame_input, placeholder_text="Path to audio file or directory")
        self.entry_in_path.drop_target_register(DND_FILES)
        self.entry_in_path.dnd_bind('<<Drop>>', self.callback_entry_in_path_dnd)
        self.entry_in_path.pack(side='top', fill = 'x', padx=10, pady=5)
        self.frame_input_config = customtkinter.CTkFrame(self.frame_input, fg_color='transparent')
        self.frame_input_config.pack(side='top')
        self.button_open_in_dir = customtkinter.CTkButton(self.frame_input_config, text='Open directory', fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=self.callback_button_open_in_path_dir)
        self.button_open_in_dir.pack(side='left', padx=10, pady=5)
        self.button_open_in_file = customtkinter.CTkButton(self.frame_input_config, text='Open file', fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=self.callback_button_open_in_path_file)
        self.button_open_in_file.pack(side='left', padx=10, pady=5)
    
        # Output
        self.label_output = customtkinter.CTkLabel(self.frame_output, text="Output model predictions")
        self.label_output.pack(side='top')
        self.entry_out_dir_path = customtkinter.CTkEntry(self.frame_output, placeholder_text="Path to directory")
        self.entry_out_dir_path.drop_target_register(DND_FILES)
        self.entry_out_dir_path.dnd_bind('<<Drop>>', self.callback_entry_out_dir_path_dnd)
        self.entry_out_dir_path.pack(side='top', fill = 'x', padx=10, pady=5)
        self.frame_output_config = customtkinter.CTkFrame(self.frame_output, fg_color='transparent')
        self.frame_output_config.pack(side='top')
        self.button_open_out_dir = customtkinter.CTkButton(self.frame_output_config, text='Open directory', fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), command=self.callback_button_open_out_path_dir)
        self.button_open_out_dir.pack(side='left', padx=10, pady=5)

        # Process
        self.switch_calc_predictions = customtkinter.CTkSwitch(self.frame_process, variable=self.use_target_model, onvalue=1, offvalue=0, command=self.callback_switch_calc_predictions, text='Calculate predictions')
        self.switch_calc_predictions.select()
        self.switch_calc_predictions.pack(side='left', padx=10, pady=5)
        self.option_out_filetype = customtkinter.CTkOptionMenu(self.frame_process, dynamic_resizing=True, values=['csv', 'Raven table'])
        self.option_out_filetype.pack(side='left', padx=10, pady=5)
        self.switch_extract_segments = customtkinter.CTkSwitch(self.frame_process, variable=self.extract_segments, onvalue=1, offvalue=0, command=self.callback_switch_extract_segments, text='Extract audio segments')
        self.switch_extract_segments.select()
        self.switch_extract_segments.pack(side='left', padx=10, pady=5)
        self.button_launch_process = customtkinter.CTkButton(self.frame_io, text='Launch process', command=self.callback_button_launch_process)
        self.button_launch_process.pack(side='bottom', fill = 'x', padx=10, pady=5)

        # Models
        self.label_model_config = customtkinter.CTkLabel(self.frame_model_config, text="Model configuration")
        self.label_model_config.pack(side='top')
        self.entry_class_labels_filepath = customtkinter.CTkEntry(self.frame_model_config, placeholder_text="Path to class labels .txt file")
        self.entry_class_labels_filepath.pack(side='top', fill = 'x', padx=10, pady=5)
        self.entry_class_labels_filepath.drop_target_register(DND_FILES)
        self.entry_class_labels_filepath.dnd_bind('<<Drop>>', self.callback_entry_class_labels_filepath_dnd)
        self.seg_button_model_selection = customtkinter.CTkSegmentedButton(self.frame_model_config, command=self.callback_seg_button_model_selection)
        self.seg_button_model_selection.pack(side='top', fill = 'x', padx=10, pady=5)
        self.seg_button_model_selection.configure(values=["Source", "Target", "Ensemble"])
        self.seg_button_model_selection.set("Source")
        self.entry_target_model_filepath = customtkinter.CTkEntry(self.frame_model_config, placeholder_text="Path to target model .tflite file")
        self.entry_target_model_filepath.pack(side='top', fill = 'x', padx=10, pady=5)
        self.entry_target_model_filepath.drop_target_register(DND_FILES)
        self.entry_target_model_filepath.dnd_bind('<<Drop>>', self.callback_entry_target_model_filepath_dnd)
        self.entry_ensemble_weights = customtkinter.CTkEntry(self.frame_model_config, placeholder_text="Path to ensemble model weights .txt file")
        self.entry_ensemble_weights.pack(side='top', fill = 'x', padx=10, pady=5)
        self.entry_ensemble_weights.drop_target_register(DND_FILES)
        self.entry_ensemble_weights.dnd_bind('<<Drop>>', self.callback_entry_ensemble_weights_dnd)
    
        # Options
        self.label_options = customtkinter.CTkLabel(self.frame_options, text="Processing options")
        self.label_options.pack(side='top')
        self.frame_threads = customtkinter.CTkFrame(self.frame_options, fg_color='transparent')
        self.frame_threads.pack(side='top', fill='x')
        self.label_options = customtkinter.CTkLabel(self.frame_threads, text="Threads")
        self.label_options.pack(side='left', padx=10)
        self.spinbox_threads = Spinbox(self.frame_threads, type='int', min=1, max=128, width=150, step_size=1)
        self.spinbox_threads.pack(side='right', padx=10, pady=5)
        self.spinbox_threads.set(8)
        self.frame_min_confidence = customtkinter.CTkFrame(self.frame_options, fg_color='transparent')
        self.frame_min_confidence.pack(side='top', fill='x')
        self.label_options = customtkinter.CTkLabel(self.frame_min_confidence, text="Min confidence")
        self.label_options.pack(side='left', padx=10)
        self.spinbox_min_confidence = Spinbox(self.frame_min_confidence, type='float', min=0.0, max=1.0, width=150, step_size=0.01)
        self.spinbox_min_confidence.pack(side='right', padx=10, pady=5)
        self.spinbox_min_confidence.set(0.0)
        self.frame_seglength = customtkinter.CTkFrame(self.frame_options, fg_color='transparent')
        self.frame_seglength.pack(side='top', fill='x')
        self.label_options = customtkinter.CTkLabel(self.frame_seglength, text="Segment length (s)")
        self.label_options.pack(side='left', padx=10)
        self.spinbox_seglength = Spinbox(self.frame_seglength, type='int', min=3, max=60, width=150, step_size=1)
        self.spinbox_seglength.pack(side='right', padx=10, pady=5)
        self.spinbox_seglength.set(3)
    
    # Menu callbacks
    def open_session_file(self):
        filepath = customtkinter.filedialog.askopenfilename()
        if filepath != '':
            with open(filepath, 'r') as f:
                session = json.load(f)
                self.entry_in_path.delete(0,customtkinter.END)
                self.entry_in_path.insert('0', session['in_path'])
                self.entry_out_dir_path.delete(0,customtkinter.END)
                self.entry_out_dir_path.insert('0', session['out_dir_path'])
                self.option_out_filetype.set(session['out_filetype'])
                if session['extract_segments']:
                    self.switch_extract_segments.select()
                else:
                    self.switch_extract_segments.deselect()
                if session['calc_predictions']:
                    self.switch_calc_predictions.select()
                else:
                    self.switch_calc_predictions.deselect()
                self.entry_class_labels_filepath.delete(0,customtkinter.END)
                self.entry_class_labels_filepath.insert('0', session['class_labels_filepath'])
                self.entry_target_model_filepath.delete(0,customtkinter.END)
                self.entry_target_model_filepath.insert('0', session['target_model_filepath'])
                self.entry_ensemble_weights.delete(0,customtkinter.END)
                self.entry_ensemble_weights.insert('0', session['ensemble_weights'])
                self.spinbox_threads.set(session['threads'])
                self.spinbox_min_confidence.set(session['min_confidence'])
                self.spinbox_seglength.set(session['seg_length'])
                self.seg_button_model_selection.set(session['model_selection'])
            print(f'Opened session file {filepath}')
    
    def save_session_file(self):
        path = customtkinter.filedialog.askdirectory()
        dialog = customtkinter.CTkInputDialog(text="Filename: (e.g. session.json)", title="Save session")
        filepath = f'{path}/{dialog.get_input()}'
        session = {
            'in_path' : self.entry_in_path.get(),
            'out_dir_path' : self.entry_out_dir_path.get(),
            'out_filetype' : self.option_out_filetype.get(),
            'calc_predictions' : self.switch_calc_predictions.get(),
            'extract_segments' : self.switch_extract_segments.get(),
            'class_labels_filepath' : self.entry_class_labels_filepath.get(),
            'model_selection' : self.seg_button_model_selection.get(),
            'target_model_filepath' : self.entry_target_model_filepath.get(),
            'ensemble_weights' : self.entry_ensemble_weights.get(),
            'threads' : self.spinbox_threads.get(),
            'min_confidence' : self.spinbox_min_confidence.get(),
            'seg_length' : self.spinbox_seglength.get()
        }
        with open(filepath, 'w') as f:
            json.dump(session, f, indent=4)
        print(f'Saved session file {filepath}')
    
    def callback_about_popup(self):
        global about_popup
        about_popup = customtkinter.CTkToplevel(self)
        about_popup.title('About')
        about_popup.geometry('1100x170')
        about_popup_textbox = customtkinter.CTkTextbox(about_popup)
        about_popup_textbox.pack(side='left', fill='both', expand=True, padx=0, pady=0)
        about_popup_textbox.insert("0.0", """
        This software is provided free and open-source under a BSD-3-Clause license. If you use it for your research, please cite as:\n\n
            Jacuzzi, G., Olden, J.D. et al. Few-shot transfer learning enables robust acoustic monitoring of wildlife communities at the landscape scale. (in preparation).\n\n
        Copyright (c) 2024, Giordano Jacuzzi.
        """)
        about_popup_textbox.configure(state="disabled")

    # Input callbacks
    def callback_entry_in_path_dnd(self, event):
        self.entry_in_path.delete(0,customtkinter.END)
        self.entry_in_path.insert('0', event.data)

    def callback_button_open_in_path_file(self):
        path = customtkinter.filedialog.askopenfilename()
        self.entry_in_path.delete(0,customtkinter.END)
        self.entry_in_path.insert('0', path)

    def callback_button_open_in_path_dir(self):
        path = customtkinter.filedialog.askdirectory()
        self.entry_in_path.delete(0,customtkinter.END)
        self.entry_in_path.insert('0', path)

    # Output callbacks
    def callback_entry_out_dir_path_dnd(self, event):
        self.entry_out_dir_path.delete(0,customtkinter.END)
        self.entry_out_dir_path.insert('0', event.data)

    def callback_button_open_out_path_dir(self):
        path = customtkinter.filedialog.askdirectory()
        self.entry_out_dir_path.delete(0,customtkinter.END)
        self.entry_out_dir_path.insert('0', path)
    
    def callback_switch_extract_segments(self):
        return

    # Model config callbacks
    def callback_seg_button_model_selection(self, value):
        return

    def callback_switch_calc_predictions(self):
        value = self.use_target_model.get()
    
    def callback_entry_class_labels_filepath_dnd(self, event):
        self.entry_class_labels_filepath.delete(0,customtkinter.END)
        self.entry_class_labels_filepath.insert('0', event.data)
    
    def callback_entry_target_model_filepath_dnd(self, event):
        self.entry_target_model_filepath.delete(0,customtkinter.END)
        self.entry_target_model_filepath.insert('0', event.data)
    
    def callback_entry_target_labels_filepath_dnd(self, event):
        self.entry_target_labels_filepath.delete(0,customtkinter.END)
        self.entry_target_labels_filepath.insert('0', event.data)

    def callback_entry_ensemble_weights_dnd(self, event):
        self.entry_ensemble_weights.delete(0,customtkinter.END)
        self.entry_ensemble_weights.insert('0', event.data)

    def callback_switch_use_ensemble(self):
        value = self.use_ensemble.get()
    
    # Options callbacks
    def callback_checkbox_retain_logit_score(self):
        return

    # Process callbacks
    def callback_button_launch_process(self):
        self.create_await_funct()
    
    def create_await_funct(self):
        threading.Thread(target=lambda loop: loop.run_until_complete(self.await_funct()),
                         args=(asyncio.new_event_loop(),)).start()
        self.button_launch_process.configure(text = "Processing...", state="disabled")
        self.progressbar.configure(mode="indeterminate")
        self.progressbar.start()

    async def await_funct(self):
        self.update_idletasks()

        in_path = self.entry_in_path.get()
        out_dir_path = self.entry_out_dir_path.get()

        out_filetype = self.option_out_filetype.get()
        extension = out_filetype
        if out_filetype == "Raven table":
            out_filetype = "table"
            extension = "txt"

        class_labels_filepath = self.entry_class_labels_filepath.get()
        
        model_selection = self.seg_button_model_selection.get()
        if model_selection == "Source":
            target_model_filepath = None
            use_ensemble = False
            ensemble_weights = None
        elif model_selection == "Target":
            target_model_filepath = self.entry_target_model_filepath.get()
            use_ensemble = False
            ensemble_weights = None
        elif model_selection == "Ensemble":
            target_model_filepath = self.entry_target_model_filepath.get()
            use_ensemble = True
            ensemble_weights = self.entry_ensemble_weights.get()

        min_confidence = self.spinbox_min_confidence.get()
        threads = self.spinbox_threads.get()

        calc_predictions = self.switch_calc_predictions.get()

        if calc_predictions:
            try:
                process_audio.process(
                    in_path                         = in_path,
                    out_dir_path                    = out_dir_path,
                    rtype                           = out_filetype,
                    target_model_filepath           = target_model_filepath,
                    slist                           = class_labels_filepath, 
                    use_ensemble                    = use_ensemble,
                    ensemble_weights                = ensemble_weights,
                    min_confidence                  = min_confidence,
                    threads                         = threads,
                    cleanup                         = True
                )
            except Exception:
                traceback.print_exc()
                print_error("Exception encountered during process, see log above")
                extract_segments = False

        extract_segments = self.extract_segments.get()
        seglength = self.spinbox_seglength.get()

        if extract_segments:
            try:
                process_audio.segment(
                    in_audio_path = in_path,
                    in_predictions_path = os.path.join(out_dir_path, 'predictions'),
                    extension = extension,
                    out_dir_path = os.path.join(out_dir_path, 'segments'),
                    min_conf = min_confidence,
                    max_segments = np.iinfo(np.int32).max, # no maximum
                    seg_length = seglength,
                    threads = threads
                )
            except Exception:
                traceback.print_exc()
                print_error("Exception encountered during segment, see log above")

        self.update_idletasks()

        self.button_launch_process.configure(text = "Launch process", state="normal")
        self.progressbar.stop()
        self.progressbar.configure(mode="determinate")
        self.progressbar.set(1.0)
        print('')

if __name__ == "__main__":
    
    freeze_support() # Freeze support for executable

    print('Launching gui...')

    customtkinter.set_appearance_mode("Dark")
    customtkinter.set_default_color_theme("green")

    app = App()
    app.mainloop()
