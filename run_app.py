"""
DeduplicationSystem Launcher
This script automatically installs needed dependencies and launches the app.
No need to build an executable - just run this with Python.
"""

import os
import sys
import subprocess
import platform

def install_dependencies():
    """Install required packages for DeduplicationSystem."""
    print("Installing required dependencies...")
    
    # Try to install dependencies
    try:
        # Upgrade pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install required packages
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "streamlit==1.26.0", 
            "pandas", 
            "plotly", 
            "pillow", 
            "imagehash", 
            "azure-storage-blob"
        ])
        
        print("All dependencies successfully installed!")
        return True
    except Exception as e:
        print(f"Error installing dependencies: {str(e)}")
        return False

def start_app():
    """Start the DeduplicationSystem app."""
    print("\nStarting DeduplicationSystem...\n")
    
    app_script = "simplified_app.py"
    if not os.path.exists(app_script):
        print(f"Error: Could not find {app_script} in current directory.")
        return False
        
    try:
        # Execute streamlit directly
        subprocess.check_call([
            sys.executable, "-m", "streamlit", "run", 
            app_script,
            "--server.headless", "true",
            "--server.port", "5000",
            "--browser.serverAddress", "localhost"
        ])
        return True
    except Exception as e:
        print(f"Error starting app: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DeduplicationSystem - One-Click Launcher")
    print("=" * 60)
    print(f"Python version: {platform.python_version()}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print("=" * 60)
    
    if install_dependencies():
        start_app()
    
    input("\nPress Enter to exit...")