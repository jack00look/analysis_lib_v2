import sys
import tkinter as tk

# Get the warning message from arguments
if len(sys.argv) > 1:
    message = sys.argv[1]
else:
    message = "Default warning!"

# Create root window
root = tk.Tk()
root.title("Warning")

# Set window size
window_width = 1000
window_height = 600

root.geometry(f"{window_width}x{window_height}")
root.resizable(False, False)

# Create a label for the warning message
label = tk.Label(root, text=message, font=("Arial", 36), fg="red", wraplength=window_width-40, justify="center")
label.pack(expand=True, fill="both", padx=20, pady=20)

# Optional: add a big OK button to close the window
button = tk.Button(root, text="OK", font=("Arial", 16), command=root.destroy)
button.pack(pady=10)

# Run the window
root.mainloop()
