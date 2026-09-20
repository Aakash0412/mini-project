import os
import sys
import platform
import psutil

def main():
    print("========================================")
    print("Environment Audit and Hardware Check")
    print("========================================")
    
    # System info
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Machine architecture: {platform.machine()}")
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    print(f"Apple Silicon: {'available' if is_apple_silicon else 'unavailable'}")
    
    # Memory
    mem = psutil.virtual_memory()
    print(f"Available system memory: {mem.available / (1024**3):.2f} GB / Total: {mem.total / (1024**3):.2f} GB")
    
    # Python version
    print(f"Python version: {sys.version.split(' ')[0]}")
    
    # Paths
    print(f"Current working directory: {os.getcwd()}")
    # Assuming script is in scripts/
    print(f"Project root: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

    print("\nPackage Versions:")
    try:
        import torch
        print(f"- PyTorch: {torch.__version__}")
    except ImportError:
        print("- PyTorch: Not installed")
        torch = None
        
    try:
        import torchvision
        print(f"- torchvision: {torchvision.__version__}")
    except ImportError:
        print("- torchvision: Not installed")
        
    try:
        import transformers
        print(f"- transformers: {transformers.__version__}")
    except ImportError:
        print("- transformers: Not installed")
        
    try:
        import sentence_transformers
        print(f"- sentence-transformers: {sentence_transformers.__version__}")
    except ImportError:
        print("- sentence-transformers: Not installed")

    print("\nDevice detection:")
    if torch:
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        print(f"- MPS: {'available' if mps_available else 'unavailable'}")
        print(f"- CPU: available")
        print("\nRecommended device:")
        recommended_device = "mps" if mps_available else "cpu"
        print(f"{'MPS' if mps_available else 'CPU'}")
        
        print("\nHardware Test:")
        try:
            device = torch.device(recommended_device)
            # Tiny PyTorch tensor test
            x = torch.ones(2, 2)
            print(f"Created tensor on CPU.")
            x = x.to(device)
            print(f"Moved tensor to {device}.")
            y = x * 2
            print(f"Performed matrix operation on {device}.")
            z = y.to("cpu")
            print(f"Moved result back to CPU: \n{z}")
            print("Hardware test successful!")
        except Exception as e:
            print(f"Hardware test failed: {e}")

if __name__ == "__main__":
    main()
