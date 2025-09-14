#!/usr/bin/env python
"""
Script to increase font sizes in existing plots
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob

def increase_font_in_plot(plot_path, output_path):
    """Increase font sizes in an existing plot"""
    try:
        # Load the image
        img = mpimg.imread(plot_path)
        
        # Create a new figure with larger font sizes
        plt.rcParams.update({
            'font.size': 16,           # Base font size
            'axes.titlesize': 18,      # Title font size
            'axes.labelsize': 16,      # Axis label font size
            'xtick.labelsize': 14,     # X-axis tick label font size
            'ytick.labelsize': 14,     # Y-axis tick label font size
            'legend.fontsize': 14,     # Legend font size
            'figure.titlesize': 20,    # Figure title font size
            'axes.linewidth': 2.0,     # Axis line width
            'grid.linewidth': 1.0,     # Grid line width
            'lines.linewidth': 2.5,    # Line width
            'patch.linewidth': 1.5     # Patch line width
        })
        
        # Create figure and display image
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        ax.axis('off')
        
        # Save with higher DPI for better quality
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"✓ Updated: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {plot_path}: {e}")
        return False

def main():
    """Process all existing plots"""
    print("Increasing font sizes in existing plots...")
    
    # Define directories to process
    plot_dirs = [
        "ml_analysis_plots/",
        "feature_analysis_plots/train/",
        "feature_analysis_plots/dev/",
        "feature_analysis_plots/test/",
        "Detecting Hallucinations_latex/"
    ]
    
    total_processed = 0
    total_success = 0
    
    for plot_dir in plot_dirs:
        if os.path.exists(plot_dir):
            print(f"\nProcessing directory: {plot_dir}")
            
            # Find all PNG files
            png_files = glob.glob(os.path.join(plot_dir, "*.png"))
            
            for png_file in png_files:
                total_processed += 1
                
                # Create backup
                backup_file = png_file + ".backup"
                if not os.path.exists(backup_file):
                    os.rename(png_file, backup_file)
                
                # Process the plot
                if increase_font_in_plot(backup_file, png_file):
                    total_success += 1
        else:
            print(f"⚠ Directory not found: {plot_dir}")
    
    print(f"\n=== Processing Complete ===")
    print(f"Total files processed: {total_processed}")
    print(f"Successfully updated: {total_success}")
    print(f"Failed: {total_processed - total_success}")

if __name__ == "__main__":
    main()
