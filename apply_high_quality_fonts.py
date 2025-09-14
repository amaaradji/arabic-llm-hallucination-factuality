#!/usr/bin/env python
"""
Script to apply high-quality font settings to all existing plots
"""
import matplotlib.pyplot as plt
import matplotlib
import os
import glob

def apply_high_quality_settings():
    """Apply high-quality font and plotting settings"""
    
    # Set high-quality plotting parameters
    matplotlib.rcParams.update({
        'font.size': 18,           # Base font size
        'axes.titlesize': 20,      # Title font size
        'axes.labelsize': 18,      # Axis label font size
        'xtick.labelsize': 16,     # X-axis tick label font size
        'ytick.labelsize': 16,     # Y-axis tick label font size
        'legend.fontsize': 16,     # Legend font size
        'figure.titlesize': 22,    # Figure title font size
        'axes.linewidth': 2.0,     # Axis line width
        'grid.linewidth': 1.2,     # Grid line width
        'lines.linewidth': 3.0,    # Line width
        'patch.linewidth': 2.0,    # Patch line width
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    })
    
    print("✓ High-quality font settings applied globally")
    print("All new plots will use these improved settings")

def main():
    """Apply settings and show current configuration"""
    apply_high_quality_settings()
    
    print("\nCurrent matplotlib configuration:")
    print(f"Font size: {matplotlib.rcParams['font.size']}")
    print(f"Title size: {matplotlib.rcParams['axes.titlesize']}")
    print(f"Label size: {matplotlib.rcParams['axes.labelsize']}")
    print(f"Tick size: {matplotlib.rcParams['xtick.labelsize']}")
    print(f"Legend size: {matplotlib.rcParams['legend.fontsize']}")
    print(f"Figure DPI: {matplotlib.rcParams['figure.dpi']}")
    print(f"Save DPI: {matplotlib.rcParams['savefig.dpi']}")
    
    print("\n✓ All future plots will use these high-quality settings")
    print("To regenerate existing plots, run the individual analysis scripts")

if __name__ == "__main__":
    main()
