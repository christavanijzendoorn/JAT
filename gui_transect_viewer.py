# -*- coding: utf-8 -*-
"""
Interactive GUI to select a transect (profile) and years, and plot cross-shore profiles.

- Load by transect ID and year range; data is fetched from the Jarkus OPeNDAP dataset via JAT.Transects

Run with:  python gui_transect_viewer.py
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Try to import toolbox (for online loading).
try:
    from JAT.Jarkus_Analysis_Toolbox import Transects
    HAS_JAT = True
except Exception:
    HAS_JAT = False

DEFAULT_JARKUS_URL = 'http://opendap.deltares.nl/thredds/dodsC/opendap/rijkswaterstaat/jarkus/profiles/transect.nc'

class TransectGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("JAT Profile Viewer")
        self.geometry("1100x700")
        self.minsize(900, 600)
        # Ensure proper teardown on window close (X button)
        self.protocol("WM_DELETE_WINDOW", self._on_exit)
        # Cache for loaded dataset and transect IDs
        self.data = None
        self.transect_ids = []
        self.filtered_transect_ids = []
        # Kustvak selection ("All" or 1..17)
        self.kustvak_var = tk.StringVar(value="All")
        # Set larger font for GUI elements
        self._set_gui_fonts()
        self._build_widgets()

    def _set_gui_fonts(self):
        """Set larger fonts for all GUI widgets."""
        default_font = ('TkDefaultFont', 13)
        label_font = ('TkDefaultFont', 13)
        button_font = ('TkDefaultFont', 13)
        entry_font = ('TkDefaultFont', 13)
        
        # Configure default fonts
        self.option_add('*TButton*Font', button_font)
        self.option_add('*TLabel*Font', label_font)
        self.option_add('*TEntry*Font', entry_font)
        self.option_add('*TCombobox*Font', entry_font)
        self.option_add('*Listbox*Font', entry_font)
        self.option_add('*TLabelframe*Font', label_font)
        
        # Create a custom style for larger labelframe labels and buttons
        style = ttk.Style()
        style.configure('Large.TLabelframe.Label', font=('TkDefaultFont', 14))
        style.configure('Large.TLabelframe', font=('TkDefaultFont', 14))
        # Make buttons larger with more padding
        style.configure('TButton', font=button_font, padding=(10, 5))

    def _build_widgets(self):
        # Top control frame
        ctrl = ttk.Frame(self, padding=8)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        # Transect browser (left) — placed first so it appears on the far left
        browser = ttk.Labelframe(ctrl, text="Transect browser", padding=8)
        browser.configure(style='Large.TLabelframe')
        browser.pack(side=tk.LEFT, fill=tk.Y, padx=(0,8))

        # Kustvak selector
        ttk.Label(browser, text="Kustvak:").grid(row=0, column=0, sticky=tk.W, padx=(0,4))
        self.kustvak_combo = ttk.Combobox(browser, width=8, state="readonly",
                                          values=["All"] + [str(i) for i in range(1, 18)],
                                          textvariable=self.kustvak_var)
        self.kustvak_combo.grid(row=0, column=1, sticky=tk.EW)
        self.kustvak_combo.bind('<<ComboboxSelected>>', self.on_kustvak_change)

        ttk.Button(browser, text="Load transects", command=self.load_transect_ids).grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(6,0))
        self.tr_listbox = tk.Listbox(browser, height=10, exportselection=False)
        self.tr_listbox.grid(row=2, column=0, columnspan=2, sticky=tk.NSEW, pady=(6,4))
        browser.rowconfigure(2, weight=1)
        browser.columnconfigure(0, weight=1)
        browser.columnconfigure(1, weight=1)
        self.tr_listbox.bind('<<ListboxSelect>>', self.on_transect_select)

        ttk.Button(browser, text="Prev", command=self.prev_transect).grid(row=3, column=0, sticky=tk.EW, padx=(0,2))
        ttk.Button(browser, text="Next", command=self.next_transect).grid(row=3, column=1, sticky=tk.EW, padx=(2,0))

        # Plot settings (to the right of browser)
        online_frame = ttk.Labelframe(ctrl, text="Plot settings", padding=8)
        online_frame.configure(style='Large.TLabelframe')
        online_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Generate year list from 1965 to 2025
        year_list = [str(y) for y in range(1965, 2026)]
        
        # Row 0: Years
        ttk.Label(online_frame, text="Start year:").grid(row=0, column=0, sticky=tk.E, padx=4)
        self.start_year_var = tk.StringVar(value="1965")
        start_year_combo = ttk.Combobox(online_frame, width=8, state="readonly", 
                                        values=year_list, textvariable=self.start_year_var)
        start_year_combo.grid(row=0, column=1, sticky=tk.W)

        ttk.Label(online_frame, text="End year (incl.):").grid(row=0, column=2, sticky=tk.E, padx=8)
        self.end_year_var = tk.StringVar(value="2025")
        end_year_combo = ttk.Combobox(online_frame, width=8, state="readonly", 
                                      values=year_list, textvariable=self.end_year_var)
        end_year_combo.grid(row=0, column=3, sticky=tk.W)

        self.load_online_btn = ttk.Button(online_frame, text="Plot (online)", command=self.plot_online)
        self.load_online_btn.grid(row=0, column=4, padx=8, rowspan=2)

        # Row 1: X-axis limits (cross-shore distance)
        ttk.Label(online_frame, text="X min:").grid(row=1, column=0, sticky=tk.E, padx=4)
        self.x_min_var = tk.StringVar(value="-500")
        ttk.Entry(online_frame, width=10, textvariable=self.x_min_var).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(online_frame, text="X max:").grid(row=1, column=2, sticky=tk.E, padx=8)
        self.x_max_var = tk.StringVar(value="100")
        ttk.Entry(online_frame, width=10, textvariable=self.x_max_var).grid(row=1, column=3, sticky=tk.W)

        # Row 2: Y-axis limits (elevation)
        ttk.Label(online_frame, text="Y min:").grid(row=2, column=0, sticky=tk.E, padx=4)
        self.y_min_var = tk.StringVar(value="-2")
        ttk.Entry(online_frame, width=10, textvariable=self.y_min_var).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(online_frame, text="Y max:").grid(row=2, column=2, sticky=tk.E, padx=8)
        self.y_max_var = tk.StringVar(value="35")
        ttk.Entry(online_frame, width=10, textvariable=self.y_max_var).grid(row=2, column=3, sticky=tk.W)

        ttk.Label(online_frame, text="(leave empty for auto)", font=('TkDefaultFont', 8), 
                  foreground='gray').grid(row=2, column=4, sticky=tk.W, padx=4)

        # Right-side utility buttons (Save, Exit)
        right_btns = ttk.Frame(ctrl)
        right_btns.pack(side=tk.RIGHT, anchor='e')
        ttk.Button(right_btns, text="Save Plot", command=self._save_plot).pack(side=tk.LEFT, padx=4)
        ttk.Button(right_btns, text="Exit", command=self._on_exit).pack(side=tk.LEFT, padx=4)

        if not HAS_JAT:
            self.load_online_btn.state(["disabled"])  # JAT not available
            ttk.Label(online_frame, foreground="red", text="JAT not installed or import failed — online plotting disabled.").grid(row=1, column=0, columnspan=5, sticky=tk.W)

        # Plot area - use constrained layout to prevent shrinking
        self.fig, self.ax = plt.subplots(figsize=(10,5), dpi=100, constrained_layout=True)
        self.ax.set_title("Cross-shore elevation profiles")
        self.ax.set_xlabel("Cross shore distance [m]")
        self.ax.set_ylabel("Elevation [m]")
        self.ax.grid(True, alpha=0.3)
        # Reserve space for colorbar (initially None)
        self.cbar = None
        self.cbar_ax = None  # Track colorbar axes

        canvas_frame = ttk.Frame(self, padding=(8,0,8,8))
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

        # Load default transect and plot after GUI is built
        self.after(100, self._load_and_plot_default)


    def _on_exit(self):
        """Close the GUI and terminate the Python process cleanly."""
        try:
            # First stop Tk event loop if running
            try:
                self.quit()
            except Exception:
                pass
            # Destroy all Tk widgets/windows
            try:
                self.destroy()
            except Exception:
                pass
            # Close any matplotlib figures
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except Exception:
                pass
            # Exit process to release terminal
            sys.exit(0)
        except SystemExit:
            raise
        except Exception:
            # Last-resort hard exit
            os._exit(0)

    def _load_and_plot_default(self):
        """Load default transect 7002800 and plot it automatically on startup."""
        if not HAS_JAT:
            return
        try:
            # Load transect IDs first
            config = {
                'name': 'jarkus',
                'years': {'start_yr': 1965, 'end_yr': 1966},  # minimal
                'transects': {'type': 'all'},
                'inputdir': '.\\Input\\',
                'outputdir': '.\\',
                'data locations': {'Jarkus': DEFAULT_JARKUS_URL},
                'save locations': { 'DirA': '', 'DirB': '', 'DirC': '', 'DirD': '', 'DirE': ''}
            }
            self.status.set("Loading transect IDs...")
            self.update_idletasks()
            self.data = Transects(config)
            ids = list(self.data.variables['id'][:])
            self.transect_ids = sorted([int(i) for i in ids])
            
            # Apply filter and populate listbox
            self.apply_kustvak_filter()
            
            # Find and select transect 7002800
            target_id = "7002800"
            for i in range(self.tr_listbox.size()):
                if self.tr_listbox.get(i) == target_id:
                    self.tr_listbox.selection_clear(0, tk.END)
                    self.tr_listbox.selection_set(i)
                    self.tr_listbox.see(i)
                    # Trigger the plot
                    self.plot_online()
                    break
        except Exception as e:
            self.status.set(f"Error loading default: {e}")

    def _save_plot(self):
        """Save the current plot to a PNG file."""
        from tkinter import filedialog
        import datetime
        
        # Get current transect ID if available
        try:
            sel = self.tr_listbox.curselection()
            if sel:
                transect_id = self.tr_listbox.get(sel[0])
            else:
                transect_id = "unknown"
        except Exception:
            transect_id = "unknown"
        
        # Generate default filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"transect_{transect_id}_{timestamp}.png"
        
        # Ask user for save location
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialfile=default_filename,
            title="Save plot as"
        )
        
        if filepath:
            try:
                self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
                self.status.set(f"Plot saved to: {filepath}")
                messagebox.showinfo("Success", f"Plot saved successfully to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot:\n{e}")
                self.status.set("Error saving plot")

    def _colormap_for_years(self, years: List[int]):
        from matplotlib import colormaps
        import matplotlib.colors as colors
        cmap = colormaps['jet']
        norm = colors.Normalize(vmin=min(years), vmax=max(years))
        return cmap, norm

    def _clear_axes(self):
        # Remove old colorbar if it exists
        if self.cbar is not None:
            self.cbar.remove()
            self.cbar = None
            self.cbar_ax = None
        self.ax.clear()
        self.ax.set_title("Cross-shore elevation profiles")
        self.ax.set_xlabel("Cross shore distance [m]")
        self.ax.set_ylabel("Elevation [m]")
        self.ax.grid(True, alpha=0.3)
        self.ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        self.ax.axhline(3, color='gray', linestyle='--', linewidth=0.6)
        self.ax.invert_xaxis()

    # --- Transect browser helpers ---
    def load_transect_ids(self):
        if not HAS_JAT:
            messagebox.showerror("Unavailable", "JAT toolbox is not available. Install the package to browse transects.")
            return
        try:
            # Build minimal config to open dataset
            config = {
                'name': 'jarkus',
                'years': {'start_yr': 1965, 'end_yr': 1966},  # minimal
                'transects': {'type': 'all'},
                'inputdir': '.\\Input\\',
                'outputdir': '.\\',
                'data locations': {'Jarkus': DEFAULT_JARKUS_URL},
                'save locations': { 'DirA': '', 'DirB': '', 'DirC': '', 'DirD': '', 'DirE': ''}
            }
            self.status.set("Loading transect IDs...")
            self.update_idletasks()
            self.data = Transects(config)
            ids = list(self.data.variables['id'][:])
            # convert numpy types to int and sort
            self.transect_ids = sorted([int(i) for i in ids])
            # apply current kustvak filter and populate listbox
            self.apply_kustvak_filter()
            total = len(self.transect_ids)
            shown = self.tr_listbox.size()
            if shown:
                self.tr_listbox.selection_set(0)
                self.tr_listbox.see(0)
            self.status.set(f"Loaded {total} transects. Showing {shown} based on Kustvak filter.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load transect IDs.\n{e}")
            self.status.set("Ready")

    def on_transect_select(self, event=None):
        try:
            sel = self.tr_listbox.curselection()
            if not sel:
                return
            # Auto-plot using current year range
            self.plot_online()
        except Exception:
            pass

    def on_kustvak_change(self, event=None):
        """Re-filter the transect list when Kustvak selection changes."""
        self.apply_kustvak_filter()
        # Auto-select first item for convenience
        if self.tr_listbox.size() > 0:
            self.tr_listbox.selection_clear(0, tk.END)
            self.tr_listbox.selection_set(0)
            self.tr_listbox.see(0)
            # Trigger plot for the first item
            self.on_transect_select()
        else:
            self._clear_axes()
            self.canvas.draw()
            self.status.set("No transects match the selected Kustvak.")

    def apply_kustvak_filter(self):
        """Filter self.transect_ids into the listbox based on selected Kustvak.

        Kustvak k (1..17) maps to first two digits of the transect ID as follows:
        - For k in 1..9, IDs start with k0 (e.g., k=7 -> '70xxxxxx')
        - For k in 10..17, IDs start with k (e.g., k=12 -> '12xxxxxx')
        """
        # Determine filter
        sel = self.kustvak_var.get()
        self.tr_listbox.delete(0, tk.END)
        if not self.transect_ids:
            self.filtered_transect_ids = []
            return
        if sel == "All" or sel.strip() == "":
            self.filtered_transect_ids = list(self.transect_ids)
        else:
            try:
                k = int(sel)
            except ValueError:
                k = None
            if k is None or not (1 <= k <= 17):
                self.filtered_transect_ids = list(self.transect_ids)
            else:
                target_prefix = k if k >= 10 else k * 10
                self.filtered_transect_ids = [tid for tid in self.transect_ids
                                              if len(str(tid)) >= 2 and int(str(tid)[:2]) == target_prefix]
        # Repopulate listbox
        for tid in self.filtered_transect_ids:
            self.tr_listbox.insert(tk.END, str(tid))

    def next_transect(self):
        sel = self.tr_listbox.curselection()
        if not sel:
            return
        idx = sel[0] + 1
        if idx >= self.tr_listbox.size():
            return
        self.tr_listbox.selection_clear(0, tk.END)
        self.tr_listbox.selection_set(idx)
        self.tr_listbox.see(idx)
        self.on_transect_select()

    def prev_transect(self):
        sel = self.tr_listbox.curselection()
        if not sel:
            return
        idx = max(0, sel[0] - 1)
        if idx == sel[0]:
            return
        self.tr_listbox.selection_clear(0, tk.END)
        self.tr_listbox.selection_set(idx)
        self.tr_listbox.see(idx)
        self.on_transect_select()

    

    def plot_online(self):
        if not HAS_JAT:
            messagebox.showerror("Unavailable", "JAT toolbox is not available. Install the package or use 'Load from file'.")
            return
        # Gather inputs
        # Get transect ID from browser selection
        sel = self.tr_listbox.curselection()
        if not sel:
            messagebox.showwarning("No transect selected", "Please load transects and select one from the list.")
            return
        try:
            transect_id = int(self.tr_listbox.get(sel[0]))
        except Exception:
            messagebox.showwarning("Input error", "Unable to parse selected transect ID.")
            return
        start_y = int(self.start_year_var.get())
        end_y_incl = int(self.end_year_var.get())
        if end_y_incl < start_y:
            messagebox.showwarning("Input error", "End year must be >= start year.")
            return

        # Build config for JAT.Transects
        config = {
            'name': 'jarkus',
            'years': {
                'start_yr': start_y,
                'end_yr': end_y_incl + 1  # end exclusive in toolbox
            },
            'transects': {
                'type': 'single',
                'single': transect_id
            },
            'inputdir': '.\\Input\\',
            'outputdir': '.\\',
            'data locations': {
                'Jarkus': DEFAULT_JARKUS_URL
            },
            'save locations': { 'DirA': '', 'DirB': '', 'DirC': '', 'DirD': '', 'DirE': ''}
        }

        try:
            self.status.set("Loading dataset...")
            self.update_idletasks()
            # Reuse loaded dataset if available, else create new
            data = self.data if self.data is not None else Transects(config)
            if self.data is None:
                self.data = data
            data.get_availability(config)
            if len(getattr(data, 'transects_filtered', [])) == 0:
                messagebox.showinfo("Not available", f"Transect {transect_id} not found in dataset.")
                return
            # Retrieve cross-shore and elevations
            cross = data.variables['cross_shore'][:]
            # indices for requested transect and years
            trs_idx = int(data.transects_filtered_idxs[0])
            year_idxs = list(map(int, data.years_filtered_idxs))
            years_avail = list(map(int, data.years_filtered))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or slice dataset.\n{e}")
            return

        # Plot
        self._clear_axes()
        # Update title with transect ID
        self.ax.set_title(f"Cross-shore elevation profiles - Transect {transect_id}")
        if years_avail:
            cmap, norm = self._colormap_for_years(years_avail)
        else:
            cmap, norm = None, None

        for yr, yr_idx in zip(years_avail, year_idxs):
            elev = data.variables['altitude'][yr_idx, trs_idx, :]
            # masked arrays may come from netCDF; turn into ndarray and filter
            elev = np.array(getattr(elev, 'filled', lambda v: elev)(np.nan))
            mask = (~np.isnan(elev)) & (elev != -9999.0)
            if np.any(mask):
                color = cmap(norm(int(yr))) if cmap is not None else 'C0'
                self.ax.plot(cross[mask], elev[mask], color=color, linewidth=1.0, alpha=0.85, label=str(yr))

        # Add colorbar for years
        if years_avail and cmap is not None and norm is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # required for ScalarMappable
            self.cbar = self.fig.colorbar(sm, ax=self.ax, label='Year', orientation='vertical', pad=0.02)

        if len(years_avail) <= 15:
            self.ax.legend(loc='best', fontsize=8, ncol=2)
        
        # Apply user-defined axis limits if provided
        try:
            x_min = self.x_min_var.get().strip()
            x_max = self.x_max_var.get().strip()
            if x_min and x_max:
                self.ax.set_xlim(float(x_max), float(x_min))  # Inverted for cross-shore
            elif x_min:
                xlim = self.ax.get_xlim()
                self.ax.set_xlim(xlim[0], float(x_min))
            elif x_max:
                xlim = self.ax.get_xlim()
                self.ax.set_xlim(float(x_max), xlim[1])
        except ValueError:
            pass  # Invalid input, ignore
        
        try:
            y_min = self.y_min_var.get().strip()
            y_max = self.y_max_var.get().strip()
            if y_min and y_max:
                self.ax.set_ylim(float(y_min), float(y_max))
            elif y_min:
                ylim = self.ax.get_ylim()
                self.ax.set_ylim(float(y_min), ylim[1])
            elif y_max:
                ylim = self.ax.get_ylim()
                self.ax.set_ylim(ylim[0], float(y_max))
        except ValueError:
            pass  # Invalid input, ignore
        
        self.canvas.draw()
        self.status.set(f"Plotted transect {transect_id} for {len(years_avail)} year(s): {years_avail[0]}–{years_avail[-1]}")


def main():
    app = TransectGUI()
    app.mainloop()
    # After the loop returns (e.g., window closed), ensure figures closed and process exits
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except Exception:
        pass
    sys.exit(0)

if __name__ == "__main__":
    main()
