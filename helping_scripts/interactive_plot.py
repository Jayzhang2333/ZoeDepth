import numpy as np
import matplotlib.pyplot as plt

class TwoSetPointEditor:
    """
    Interactive editor showing two point sets:
      - Editable grid points (in red), automatically filtering out border points
      - Fixed sparse points (in green)
    Controls:
      • Left-click + drag on red point to move it.
      • Double-click on red point to remove it.
      • Shift + left-click on empty space to add a new red point.
      • Press 's' to save edited red points and exit.
    Border points (x or y == -1 or 1) are automatically removed.
    """
    def __init__(self, grid_npy_path, sparse_npy_path, save_path="edited_grid.npy"):
        # Load first set (grid) from (H,W,2) with (y,x) coords
        grid = np.load(grid_npy_path)                # shape (H,W,2)
        pts = grid.reshape(-1, 2)                    # (N,2) as (y,x)
        self.grid_pts = pts[:, [1,0]].tolist()       # convert to (x,y)
        
        # Filter out border points
        self.grid_pts = [
            [x, y] for x, y in self.grid_pts
            if -1 < x < 1 and -1 < y < 1
        ]

        # Load second set (sparse) from (M,3) with (x,y,depth)
        sparse = np.load(sparse_npy_path)            # shape (M,3)
        self.sparse_pts = sparse[:, :2].tolist()     # (x,y) only

        self.save_path = save_path

        # Setup plot
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        # Sparse points (green)
        self.ax.scatter(
            [p[0] for p in self.sparse_pts],
            [p[1] for p in self.sparse_pts],
            c='green', s=40, label='sparse (fixed)'
        )
        # Editable grid points (red)
        self.scatter = self.ax.scatter(
            [p[0] for p in self.grid_pts],
            [p[1] for p in self.grid_pts],
            c='red', s=40, picker=5, label='grid (editable)'
        )

        self.ax.set_xlim(-1,1)
        self.ax.set_ylim(-1,1)
        self.ax.invert_yaxis()
        self.ax.set_xlabel('x (normalized)')
        self.ax.set_ylabel('y (normalized)')
        self.ax.set_title("Red = editable | Green = fixed\nShift+click to add, double-click to remove, drag to move, 's' to save")
        self.ax.legend(loc='upper right')

        # Interaction state
        self.selected = None
        # Connect events
        self.cid_press   = self.fig.canvas.mpl_connect('button_press_event',   self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion  = self.fig.canvas.mpl_connect('motion_notify_event',  self.on_motion)
        self.cid_key     = self.fig.canvas.mpl_connect('key_press_event',      self.on_key)

        plt.show()

    def on_press(self, event):
        if event.inaxes != self.ax: 
            return
        # Add new red point: Shift + left-click empty
        if event.button == 1 and event.key == 'shift':
            x, y = event.xdata, event.ydata
            if -1 < x < 1 and -1 < y < 1:
                self.grid_pts.append([x, y])
                self.update()
            return
        # Check if clicked on red point
        contains, info = self.scatter.contains(event)
        if contains:
            idx = info['ind'][0]
            # Remove on double-click
            if event.dblclick:
                self.grid_pts.pop(idx)
                self.update()
            else:
                # Begin dragging
                self.selected = idx

    def on_motion(self, event):
        if self.selected is None or event.inaxes != self.ax:
            return
        # Move selected red point
        x, y = event.xdata, event.ydata
        self.grid_pts[self.selected] = [x, y]
        self.update()

    def on_release(self, event):
        # End dragging and filter border
        if self.selected is not None:
            self.selected = None
            self.grid_pts = [
                [x_, y_] for x_, y_ in self.grid_pts
                if -1 < x_ < 1 and -1 < y_ < 1
            ]
            self.update()

    def on_key(self, event):
        if event.key == 's':
            # Save only red (grid) points
            pts_arr = np.array(self.grid_pts)  # shape (N,2) as (x,y)
            np.save(self.save_path, pts_arr)
            print(f"Saved {len(self.grid_pts)} grid points to {self.save_path}")
            plt.close(self.fig)

    def update(self):
        # Filter border before update
        self.grid_pts = [
            [x, y] for x, y in self.grid_pts
            if -1 < x < 1 and -1 < y < 1
        ]
        coords = np.array(self.grid_pts)
        self.scatter.set_offsets(coords)
        self.fig.canvas.draw_idle()

# === Usage ===
# Replace the paths below with your actual .npy files
# grid_npy_path should be (H, W, 2) array of normalized (y,x)
# sparse_npy_path should be (M, 3) array of [x, y, depth]
editor = TwoSetPointEditor(
    grid_npy_path="top_left_deform_demo.npy",
    sparse_npy_path="/home/jay/ZoeDepth/top_left_sparse_depth.npy",
    save_path="top_left_edited_grid.npy"
)