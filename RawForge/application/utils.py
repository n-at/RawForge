import onnxruntime as ort

def get_best_providers():
    available = ort.get_available_providers()
    
    # The "Hierarchy of Speed"
    priority = [
        "CUDAExecutionProvider",      # NVIDIA
        "ROCMExecutionProvider",      # AMD (Direct)
        "MIGraphXExecutionProvider",  # AMD (Optimized)
        "CoreMLExecutionProvider",    # Apple Silicon
        "DmlExecutionProvider",       # Windows (AMD/Intel/Generic GPU)
        "CPUExecutionProvider"        # The fallback
    ]
    
    return [p for p in priority if p in available]