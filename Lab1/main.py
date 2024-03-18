import matplotlib.pyplot as plt
import numpy as np
import fourier as f
import tkinter as tk
from tkinter import Label, Entry, Button
from pathlib import Path


class FourierVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Fourier Series Visualizer")

        # Default bounds and precision
        self.lower_bound = -3.13
        self.upper_bound = 3.13
        self.iterations = 50

        # Entry widgets for user input
        self.lower_bound_label = Label(master, text="Lower Bound:")
        self.lower_bound_entry = Entry(master)
        self.lower_bound_entry.insert(0, str(self.lower_bound))

        self.upper_bound_label = Label(master, text="Upper Bound:")
        self.upper_bound_entry = Entry(master)
        self.upper_bound_entry.insert(0, str(self.upper_bound))

        self.iterations_label = Label(master, text="Iterations:")
        self.iterations_entry = Entry(master)
        self.iterations_entry.insert(0, str(self.iterations))

        self.function_choice_label = Label(master, text="Choose a function:")
        self.function_choice = tk.StringVar()
        self.function_choice.set("f")
        self.function_choice_f = tk.Radiobutton(
            master, text="f(x)", variable=self.function_choice, value="f")
        self.function_choice_f_2 = tk.Radiobutton(
            master, text="f_2(x)", variable=self.function_choice, value="f_2")

        # Button to trigger visualization
        self.visualize_button = Button(
            master, text="Visualize", command=self.visualize)

        # Positioning widgets
        self.lower_bound_label.grid(row=0, column=0, padx=10, pady=5)
        self.lower_bound_entry.grid(row=0, column=1, padx=10, pady=5)
        self.upper_bound_label.grid(row=1, column=0, padx=10, pady=5)
        self.upper_bound_entry.grid(row=1, column=1, padx=10, pady=5)
        self.iterations_label.grid(row=2, column=0, padx=10, pady=5)
        self.iterations_entry.grid(row=2, column=1, padx=10, pady=5)
        self.function_choice_label.grid(row=3, column=0, padx=10, pady=5)
        self.function_choice_f.grid(row=4, column=0, padx=10, pady=5)
        self.function_choice_f_2.grid(row=4, column=1, padx=10, pady=5)
        self.visualize_button.grid(row=5, column=0, columnspan=2, pady=10)

    def visualize(self):
        # Get user input values
        self.lower_bound = float(self.lower_bound_entry.get())
        self.upper_bound = float(self.upper_bound_entry.get())
        self.iterations = int(self.iterations_entry.get())

        # Generate x values based on user input
        x_values = np.linspace(self.lower_bound, self.upper_bound, 1000)
        # original_values = [f.f(x) for x in x_values]
        original_values = [f.f(x) if self.function_choice.get(
        ) == "f" else f.f_2(x) for x in x_values]

        p = Path(__file__).with_name('fourier.log')
        with (p.open('w')) as file:
            if self.function_choice.get() == "f":
                fourier_values, error, quad_error = f.fourier_range(
                    x_values, self.iterations, file)
            else:
                fourier_values, error, quad_error = f.fourier_range_2(
                    x_values, self.iterations, file)

        # Plotting the original function
        plt.plot(x_values, original_values, label='Original Function')

        # Plotting the Fourier series approximation
        plt.plot(x_values, fourier_values,
                 label='Fourier Series Approximation')

        ax = plt.gca()

        y_margin = 0.15 * max(np.absolute(original_values))
        ax.set_ylim(min(original_values) - y_margin,
                    max(original_values) + y_margin)

        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.suptitle(
            f'Visualization of Original Function and Fourier Series Approximation')
        plt.title(
            f"N={self.iterations}\nAverage {'Absolute' if self.function_choice == 'f' else 'Relative'} Error: {error:.4f}\nQuadratic error: {quad_error:.4f}")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = FourierVisualizer(root)
    root.mainloop()
