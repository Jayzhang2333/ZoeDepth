import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mplcursors  # pip install mplcursors
import tifffile   # pip install tifffile

def main():
    # Set the file paths for the image and CSV file
    index = '16315111131960237'
    image_path = '/media/jay/Marine1/archive/red_sea/northeast_path/imgs/'+index+'.tiff'
    csv_path = '/media/jay/Marine1/archive/red_sea/northeast_path/sparse_depth/'+index+'_sparse_depth.csv'
    
    # Load the image and CSV file.
    image = plt.imread(image_path)
    df = pd.read_csv(csv_path)  # Expected columns: 'row', 'column', 'depth'
    
    # Filter out any points deeper than 10 m
    df = df[df['depth'] <= 10.0].copy()
    
    # Extract pixel coords and (clamped) depths
    x = df['column']
    y = df['row']
    depth = df['depth']
    
    # Recompute total number of points
    total_points = len(df)
    print(f"Total number of points (≤10 m): {total_points}")
    
    # Build a full-size depth matrix, masking values >10 m
    height, width = image.shape[:2]
    depth_matrix = np.full((height, width), np.nan, dtype=np.float32)
    for _, row_data in df.iterrows():
        r, c = int(row_data['row']), int(row_data['column'])
        depth_matrix[r, c] = row_data['depth']
    
    # Optional: save out the masked depth TIFF
    # image_dir = os.path.dirname(image_path)
    # base_name = os.path.splitext(os.path.basename(image_path))[0]
    # output_path = os.path.join(image_dir, f"depth_{base_name}_clipped.tif")
    # tifffile.imwrite(output_path, depth_matrix)
    # print(f"Clipped depth matrix saved as '{output_path}'")
    
    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # 1) Overlay scatter of ≤10 m points
    ax1 = axs[0]
    ax1.imshow(image, origin='upper')
    scatter = ax1.scatter(
        x, y, c=depth, cmap='viridis', s=30,
        vmin=0, vmax=10
    )
    ax1.set_title('Overlay Image with Depth ≤10 m')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.text(0.05, 0.95, f"Points ≤10 m: {total_points}",
             transform=ax1.transAxes, fontsize=12, color='white',
             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    
    mplcursors.cursor(scatter, hover=True).connect(
        "add",
        lambda sel: sel.annotation.set_text(
            f"Depth: {depth.iloc[sel.index]:.2f} m\nx: {x.iloc[sel.index]:.0f}, y: {y.iloc[sel.index]:.0f}"
        )
    )
    
    # 2) Depth matrix image, clipped at 10 m
    ax2 = axs[1]
    im = ax2.imshow(
        depth_matrix, origin='upper', cmap='viridis',
        vmin=0, vmax=10
    )
    ax2.set_title('Depth Matrix (clipped to 10 m)')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label('Depth (m)')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
