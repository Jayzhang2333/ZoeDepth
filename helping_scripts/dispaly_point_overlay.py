import matplotlib.pyplot as plt
import csv
from PIL import Image
import numpy as np
import mplcursors

def display_sparse_overlay_interactive(
    image_path: str,
    csv_path: str,
    scatter_size: int = 50,
    cmap: str = 'Reds',
    alpha: float = 0.8
):
    """
    Load an RGB image and a CSV of (row, column) points, then overlay the points on the image.
    Hovering over a point will show its (row, column) pixel coordinates.
    """
    # Load image
    img = np.array(Image.open(image_path))

    # Read points
    rows, cols = [], []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for entry in reader:
            rows.append(int(entry['row']))
            cols.append(int(entry['column']))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    sc = ax.scatter(cols, rows,
                    c=rows,       # or any array of same length; color isn't used for hover
                    cmap=cmap,
                    s=scatter_size,
                    alpha=alpha,
                    edgecolors='black')

    ax.set_axis_off()
    ax.set_title('RGB Image with Sparse Points (hover to see coords)')

    # Enable hover annotations
    cursor = mplcursors.cursor(sc, hover=True)
    @cursor.connect("add")
    def _(sel):
        x, y = sel.target  # x=col, y=row
        sel.annotation.set_text(f"row={int(round(y))}, col={int(round(x))}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

    plt.show()


display_sparse_overlay_interactive('/home/jay/Downloads/first_100_cropped_resized/00002.png', '/home/jay/ZoeDepth/turbid_points.csv')