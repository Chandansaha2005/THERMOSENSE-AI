"""
THERMOSENSE-AI Automated Setup Script
Handles complete project setup and launch
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def run_command(description, command):
    """Run command and handle errors"""
    print(f"📌 {description}")
    print(f"   Command: {' '.join(command)}")
    
    try:
        subprocess.run(command, check=True)
        print(f"   ✅ Success\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}\n")
        return False
    except FileNotFoundError:
        print(f"   ❌ Command not found\n")
        return False


def create_directories():
    """Create project directories"""
    print_header("Creating Directories")
    
    directories = ['data', 'models', 'simulation', 'ml', 'controller', 'dashboard', 'utils']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ {directory}/")


def check_python():
    """Check Python version"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"   Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("   ❌ Python 3.10+ required")
        return False
    
    print("   ✅ Version OK")
    return True


def install_dependencies():
    """Install packages"""
    print_header("Installing Dependencies")
    
    packages = [
        'numpy>=1.24.3',
        'pandas>=2.0.3',
        'scikit-learn>=1.3.0',
        'streamlit>=1.28.0',
        'matplotlib>=3.7.2',
        'plotly>=5.17.0',
        'joblib>=1.3.2'
    ]
    
    for package in packages:
        if not run_command(f"Installing {package}", 
                          [sys.executable, '-m', 'pip', 'install', '-q', package]):
            return False
    
    return True


def generate_data():
    """Generate simulation data"""
    print_header("Generating Sensor Data")
    
    if os.path.exists('data/simulated_data.csv'):
        response = input("   Data exists. Regenerate? (y/n): ")
        if response.lower() != 'y':
            print("   ⏭️  Skipping\n")
            return True
    
    return run_command("Generating data", [sys.executable, 'simulation/sensors.py'])


def train_models():
    """Train ML models"""
    print_header("Training ML Models")
    
    models_exist = (
        os.path.exists('models/occupancy_model.pkl') and
        os.path.exists('models/temperature_model.pkl')
    )
    
    if models_exist:
        response = input("   Models exist. Retrain? (y/n): ")
        if response.lower() != 'y':
            print("   ⏭️  Skipping\n")
            return True
    
    print("\n   [1/2] Occupancy Model...")
    if not run_command("Training occupancy", [sys.executable, 'ml/occupancy_model.py']):
        return False
    
    print("   [2/2] Temperature Model...")
    if not run_command("Training temperature", [sys.executable, 'ml/heat_model.py']):
        return False
    
    return True


def launch_dashboard():
    """Launch Streamlit"""
    print_header("Launching Dashboard")
    
    print("   🚀 Starting Streamlit...")
    print("   📱 Opens in browser")
    print("   🛑 Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dashboard/app.py'])
    except KeyboardInterrupt:
        print("\n\n   Dashboard stopped")


def main():
    """Main setup flow"""
    print("\n" + "=" * 60)
    print("  🌡️  THERMOSENSE-AI - Automated Setup")
    print("=" * 60)
    
    # Check Python
    if not check_python():
        print("\n❌ Setup failed: Python version")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    print("\n")
    response = input("📦 Install dependencies? (y/n): ")
    if response.lower() == 'y':
        if not install_dependencies():
            print("\n❌ Installation failed")
            sys.exit(1)
    
    # Generate data
    print("\n")
    response = input("📊 Generate data? (y/n): ")
    if response.lower() == 'y':
        if not generate_data():
            print("\n❌ Data generation failed")
            sys.exit(1)
    
    # Train models
    print("\n")
    response = input("🧠 Train models? (y/n): ")
    if response.lower() == 'y':
        if not train_models():
            print("\n❌ Training failed")
            sys.exit(1)
    
    # Launch dashboard
    print("\n")
    response = input("🚀 Launch dashboard? (y/n): ")
    if response.lower() == 'y':
        launch_dashboard()
    else:
        print("\n✅ Setup complete!")
        print("\nTo launch later:")
        print(f"   {sys.executable} -m streamlit run dashboard/app.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)