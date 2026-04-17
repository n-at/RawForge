MODEL_REGISTRY = {
    "XFormerDenoise": {
        "url": "https://github.com/rymuelle/RawForge/releases/download/onnx_v1.0.0/xformer_static.onnx",
        "filename": "xformer_static.onnx",
        "backend": "rawpy",
        "conditioning": "false",
        "batch_size": 1, 
        "crop_size": 256,
    }, 
    "NAFDenoise": {
        "url": "https://github.com/rymuelle/RawForge/releases/download/onnx_v1.0.0/NAF_static.onnx",
        "filename": "NAF_static.onnx",
        "backend": "rawpy",
        "conditioning": "false",
        "batch_size": 1, 
        "crop_size": 256,
    }, 
}
