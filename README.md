# Media Deduplication System

A powerful application to deduplicate media files in your local storage and migrate unique files to Azure Blob Storage.

![DeduplicationSystem](generated-icon.png)

## Features

- **Multiple Detection Methods**: Combines perceptual hashing with SHA-512
- **Smart Processing**: Efficiently handles large collections of media files
- **Detailed Results**: Comprehensive statistics and visualizations
- **Azure Integration**: Seamless migration to cloud storage
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Download & Installation

### Option 1: Run as a Standalone App (Recommended)

1. Download the complete code
2. Run the appropriate setup script for your system:
   - Windows: Double-click `setup.bat`
   - macOS/Linux: Run `./setup.sh` in a terminal
3. Follow the instructions in the setup script
4. After setup completes, find the executable in the `dist` folder
5. You can copy the `dist` folder to any computer and run the app without installing Python

### Option 2: Run with Python

1. Download the complete code
2. Install Python 3.8 or higher
3. Install required dependencies:
   ```
   pip install streamlit pandas plotly pillow imagehash azure-storage-blob
   ```
4. Run the app:
   ```
   streamlit run simplified_app.py
   ```

## How to Use

### Local Storage Paths

- **Windows**: Use paths like `D:\Photos` or `C:\Users\YourName\Pictures`
- **macOS/Linux**: Use paths like `/Users/YourName/Pictures` or `/home/YourName/Pictures`

### Azure Blob Storage Configuration

To enable Azure integration, you'll need:
- Azure Key
- Connection String
- Container Name

These can be obtained from your Azure portal.

## Process Flow

1. Configure local storage path and other settings
2. Click "Start Process" to begin deduplication
3. Review detailed results and statistics
4. Optionally migrate unique files to Azure

## Support

For questions or support, please contact your IT administrator.