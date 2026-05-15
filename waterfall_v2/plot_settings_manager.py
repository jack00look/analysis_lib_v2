"""
Plot Settings Manager for waterfall_v2

Manages persistent plot configurations (axis limits, colormap ranges, etc.) indexed by plot name/ID
instead of figure number. This ensures settings are preserved when the number of plots changes.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, Callable


class PlotSettingsManager:
    """
    Manages plot-specific settings (axis ranges, colormaps, etc.) indexed by plot ID.
    
    Settings are stored in a JSON file and persist across sessions. This prevents
    plot configurations from being lost when the number of dynamically-created plots changes.
    
    Usage:
        manager = PlotSettingsManager()
        
        # Create a named figure
        fig, ax = plt.subplots()
        manager.register_figure('fluct_waterfall', fig)
        
        # Apply previously saved settings if they exist
        manager.apply_settings('fluct_waterfall', ax)
        
        # Enable auto-save on user interactions
        manager.enable_autosave('fluct_waterfall', ax)
        
        # After user modifies plot via GUI, settings are automatically saved
    """
    
    DEFAULT_SETTINGS_FILE = '.waterfall_plot_settings.json'
    
    def __init__(self, settings_dir: Optional[str] = None):
        """
        Initialize the plot settings manager.
        
        Parameters
        ----------
        settings_dir : str, optional
            Directory to store settings file. Defaults to current directory.
        """
        if settings_dir is None:
            settings_dir = os.getcwd()
        
        self.settings_dir = Path(settings_dir)
        self.settings_file = self.settings_dir / self.DEFAULT_SETTINGS_FILE
        self.settings = self._load_settings()
        self.registered_figures = {}  # Map plot_id -> fig object
        self.autosave_callbacks = {}  # Map plot_id -> callback objects
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from JSON file if it exists."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load plot settings from {self.settings_file}: {e}")
                return {}
        return {}
    
    def _save_settings(self) -> None:
        """Save settings to JSON file."""
        try:
            os.makedirs(self.settings_dir, exist_ok=True)
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save plot settings to {self.settings_file}: {e}")
    
    def register_figure(self, plot_id: str, fig) -> None:
        """
        Register a figure with a specific ID.
        
        Parameters
        ----------
        plot_id : str
            Unique identifier for the plot (e.g., 'fluct_waterfall', 'evol_analysis')
        fig : matplotlib.figure.Figure
            The matplotlib figure object
        """
        self.registered_figures[plot_id] = fig
    
    def apply_settings(self, plot_id: str, ax: plt.Axes) -> bool:
        """
        Apply saved settings to an axes object.
        
        Parameters
        ----------
        plot_id : str
            The plot ID to retrieve settings for
        ax : matplotlib.axes.Axes
            The axes object to apply settings to
        
        Returns
        -------
        bool
            True if settings were applied, False if no settings found
        """
        if plot_id not in self.settings:
            return False
        
        settings = self.settings[plot_id]
        
        # Apply axis limits
        if 'xlim' in settings and settings['xlim'] is not None:
            try:
                ax.set_xlim(settings['xlim'])
            except Exception as e:
                print(f"Warning: Could not set xlim for {plot_id}: {e}")
        
        if 'ylim' in settings and settings['ylim'] is not None:
            try:
                ax.set_ylim(settings['ylim'])
            except Exception as e:
                print(f"Warning: Could not set ylim for {plot_id}: {e}")
        
        return True
    
    def save_settings(self, plot_id: str, ax: plt.Axes) -> None:
        """
        Save current axes settings to persistent storage.
        
        Parameters
        ----------
        plot_id : str
            Unique identifier for the plot
        ax : matplotlib.axes.Axes
            The axes object to extract settings from
        """
        if plot_id not in self.settings:
            self.settings[plot_id] = {}
        
        # Save axis limits
        try:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            self.settings[plot_id]['xlim'] = list(xlim) if xlim is not None else None
            self.settings[plot_id]['ylim'] = list(ylim) if ylim is not None else None
        except Exception as e:
            print(f"Warning: Could not extract axes settings from {plot_id}: {e}")
        
        self._save_settings()
    
    def enable_autosave(self, plot_id: str, ax: plt.Axes) -> None:
        """
        Enable automatic saving of plot settings when user interacts with the plot.
        
        This will save settings whenever the plot is modified through the GUI.
        
        Parameters
        ----------
        plot_id : str
            The plot ID
        ax : matplotlib.axes.Axes
            The axes object to monitor
        """
        if plot_id in self.autosave_callbacks:
            # Already enabled for this plot
            return
        
        fig = ax.figure
        
        # Create a callback that saves settings on any figure event
        def on_figure_changed(event):
            try:
                self.save_settings(plot_id, ax)
            except Exception as e:
                # Silently ignore errors to avoid flooding output during interaction
                pass
        
        # Connect to multiple events that indicate user interaction
        cids = []
        try:
            # These events fire when user modifies axis limits
            cids.append(fig.canvas.mpl_connect('button_release_event', on_figure_changed))
            cids.append(fig.canvas.mpl_connect('scroll_event', on_figure_changed))
            cids.append(fig.canvas.mpl_connect('key_release_event', on_figure_changed))
        except Exception as e:
            print(f"Warning: Could not enable autosave callbacks for {plot_id}: {e}")
            return
        
        # Store callback IDs for potential disconnection later
        self.autosave_callbacks[plot_id] = {
            'figure': fig,
            'callback_ids': cids,
            'on_figure_changed': on_figure_changed,
        }
        
        print(f"Plot settings autosave enabled for '{plot_id}'")
    
    def disable_autosave(self, plot_id: str) -> None:
        """
        Disable automatic saving for a specific plot.
        
        Parameters
        ----------
        plot_id : str
            The plot ID to disable autosave for
        """
        if plot_id not in self.autosave_callbacks:
            return
        
        callbacks = self.autosave_callbacks[plot_id]
        fig = callbacks['figure']
        
        for cid in callbacks['callback_ids']:
            try:
                fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        
        del self.autosave_callbacks[plot_id]
    
    def clear_settings(self, plot_id: Optional[str] = None) -> None:
        """
        Clear saved settings for a plot or all plots.
        
        Parameters
        ----------
        plot_id : str, optional
            If provided, clears settings for only this plot.
            If None, clears all settings.
        """
        if plot_id is None:
            self.settings = {}
        elif plot_id in self.settings:
            del self.settings[plot_id]
        
        self._save_settings()
    
    def get_settings(self, plot_id: str) -> Dict[str, Any]:
        """
        Get all saved settings for a plot.
        
        Parameters
        ----------
        plot_id : str
            The plot ID
        
        Returns
        -------
        dict
            The settings dictionary, or empty dict if not found
        """
        return self.settings.get(plot_id, {})
    
    def list_saved_plots(self) -> list:
        """
        List all plot IDs that have saved settings.
        
        Returns
        -------
        list
            List of plot IDs
        """
        return list(self.settings.keys())


# Global instance (convenient for single-threaded use)
_global_manager = None


def get_plot_manager(settings_dir: Optional[str] = None) -> PlotSettingsManager:
    """
    Get or create the global plot settings manager instance.
    
    Parameters
    ----------
    settings_dir : str, optional
        Directory to store settings. Only used on first call.
    
    Returns
    -------
    PlotSettingsManager
        The global manager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = PlotSettingsManager(settings_dir)
    return _global_manager
