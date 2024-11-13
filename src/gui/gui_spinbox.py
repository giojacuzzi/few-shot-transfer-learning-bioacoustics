import customtkinter
from typing import Callable, Union
import numpy as np

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

        self.configure(fg_color=("gray78", "gray28"))  # set frame color

        self.grid_columnconfigure((0, 2), weight=0)  # buttons don't expand
        self.grid_columnconfigure(1, weight=1)  # entry expands

        self.subtract_button = customtkinter.CTkButton(self, text="-", width=height-6, height=height-6,
                                                       command=self.subtract_button_callback)
        self.subtract_button.grid(row=0, column=0, padx=(3, 0), pady=3)

        self.entry = customtkinter.CTkEntry(self, justify=customtkinter.CENTER, width=width-(2*height), height=height-6, border_width=0)
        self.entry.grid(row=0, column=1, columnspan=1, padx=3, pady=3, sticky="ew")

        self.add_button = customtkinter.CTkButton(self, text="+", width=height-6, height=height-6,
                                                  command=self.add_button_callback)
        self.add_button.grid(row=0, column=2, padx=(0, 3), pady=3)

        # default value
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
            print(self.step_size)
    
def float_to_str(value: float, step_size: float):
    value = float(value)
    digits = len(str(step_size).split('.')[-1])
    return(f"{round(value, digits):.{digits}f}")
