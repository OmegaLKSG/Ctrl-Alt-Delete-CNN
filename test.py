import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to update the pie chart size when window is resized
def resize_pie_chart(event, fig, canvas):
    width = event.width / 100  # Scale down to match the figure size
    height = event.height / 100
    fig.set_size_inches(width, height)  # Set the new figure size
    canvas.draw()
# Function to create the pie chart and embed it in the CustomTkinter frame
def create_pie_chart(frame):
    labels = ['Category A', 'Category B', 'Category C', 'Category D']
    sizes = [20, 30, 25, 25]
    colors = ['#005eff', '#002d7a', '#000b61', '#3b007a']
    explode = (0, 0, 0, 0)

    # Create a matplotlib figure and set background color
    fig, ax = plt.subplots(facecolor='#333333')  # Figure background color
    ax.set_facecolor('#333333') # Axes background color

    wedges, texts, autotexts = ax.pie(sizes, 
                                      explode=explode, 
                                      labels=labels, 
                                      colors=colors,
                                      autopct='%1.1f%%', 
                                      shadow=True, 
                                      startangle=140,
                                      textprops={'color': 'white'})
    
    for autotext in autotexts:
        autotext.set_color('white')
    ax.axis('equal')

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    # dynamically resize the chart
    frame.bind("<Configure>", lambda event: resize_pie_chart(event, fig, canvas))
# Initialize CustomTkinter app
app = ctk.CTk()
app.geometry("500x500")
app.title("Dynamically Resizing Pie Chart in CustomTkinter")

# Create a CustomTkinter frame
frame = ctk.CTkFrame(master=app)
frame.pack(pady=20, padx=20, fill="both", expand=True)
frame.pack_propagate(False)  # Allow frame to control resizing

# Create the pie chart inside the frame
create_pie_chart(frame)

# Start the app
app.mainloop()
