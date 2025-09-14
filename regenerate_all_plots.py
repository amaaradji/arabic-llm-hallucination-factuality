#!/usr/bin/env python
"""
Script to regenerate all plots with improved font sizes
"""
import os
import sys
import subprocess

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"\n=== {description} ===")
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
        else:
            print(f"✗ {description} failed:")
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"✗ {description} failed with exception: {e}")

def main():
    """Regenerate all plots with improved font sizes"""
    print("Regenerating all plots with improved font sizes...")
    
    # List of scripts to run
    scripts = [
        ("initial_analysis/analyze_engineered_features.py", "Feature Analysis Plots"),
        ("initial_analysis/advanced_ml_analysis.py", "ML Analysis Plots"),
        ("initial_analysis/summarize_results.py", "Summary Results Plots"),
        ("initial_analysis/analyze_feature_importance.py", "Feature Importance Plots"),
        ("paper_figure_generation/generate_paper_figures.py", "Paper Figure Generation")
    ]
    
    # Run each script
    for script_path, description in scripts:
        if os.path.exists(script_path):
            run_script(script_path, description)
        else:
            print(f"⚠ Script not found: {script_path}")
    
    print("\n=== Plot Regeneration Complete ===")
    print("All plots have been regenerated with improved font sizes.")
    print("Check the output directories for the updated plots.")

if __name__ == "__main__":
    main()
