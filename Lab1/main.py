import matplotlib.pyplot as plt
import numpy as np
import fourier as f
import tkinter as tk
from tkinter import Label, Entry, Button


class FourierVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Fourier Series Visualizer")

        # Default bounds and precision
        self.lower_bound = -10
        self.upper_bound = 10
        self.iterations = 50
        self.file = open("fourier_series.log", "w")

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
        self.visualize_button.grid(row=3, column=0, columnspan=2, pady=10)

    def visualize(self):
        # Get user input values
        self.lower_bound = float(self.lower_bound_entry.get())
        self.upper_bound = float(self.upper_bound_entry.get())
        self.iterations = int(self.iterations_entry.get())

        # Generate x values based on user input
        x_values = np.linspace(self.lower_bound, self.upper_bound, 1000)

        original_values = [f.f(x) for x in x_values]
        fourier_values = f.fourier_range(x_values, self.iterations, self.file)
        self.file.close()
        # Plotting the original function
        plt.plot(x_values, original_values, label='Original Function')

        # Plotting the Fourier series approximation
        plt.plot(x_values, fourier_values,
                 label='Fourier Series Approximation')

        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.suptitle(
            f'Visualization of Original Function and Fourier Series Approximation')
        plt.title(
            f"N={self.iterations}\nAverage Absolute Error: {f.absolute_error(original_values, fourier_values):.4f}")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = FourierVisualizer(root)
    root.mainloop()
