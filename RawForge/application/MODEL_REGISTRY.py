MODEL_REGISTRY = {
    "TreeNetDenoise": {
        "url": "https://github.com/rymuelle/RawRefinery/releases/download/v1.2.1-alpha/ShadowWeightedL1.pt",
        "filename": "ShadowWeightedL1.pt"
    },
    "TreeNetDenoiseLight": {
        "url": "    https://github.com/rymuelle/RawRefinery/releases/download/v1.2.1-alpha/ShadowWeightedL1_light.pt",
        "filename": "ShadowWeightedL1_light.pt"
    },
    "TreeNetDenoiseSuperLight": {
        "url": "https://github.com/rymuelle/RawRefinery/releases/download/v1.2.1-alpha/ShadowWeightedL1_super_light.pt",
        "filename": "ShadowWeightedL1_super_light.pt"
    },
    "TreeNetDenoiseHeavy": {
        "url": "https://github.com/rymuelle/RawRefinery/releases/download/v1.2.1-alpha/ShadowWeightedL1_24_deep_500.pt",
        "filename": "ShadowWeightedL1_24_deep_500.pt"
    },
    "Deblur": {
        "url": "https://github.com/rymuelle/RawRefinery/releases/download/v1.2.1-alpha/realblur_gamma_140.pt",
        "filename": "realblur_gamma_140.pt",
        "affine": True,
    },
    "DeepSharpen": {
        "url": "https://github.com/rymuelle/RawRefinery/releases/download/v1.2.1-alpha/Deblur_deep_24.pt",
        "filename": "Deblur_deep_24.pt",
        "affine": True,
    },
    "test_run_baseline.pt": {
        "url": None,
        "filename": "test_run_baseline.pt",
        "affine": True,
    },
    "TreeNetDenoiseXTrans": {
        "url": "https://github.com/rymuelle/RawForge/releases/download/xtrans_v1.0.0/xtrans_fixed_exposure_no_conditioning_400.pt",
        "filename": "xtrans_fixed_exposure_no_conditioning_400.pt",
        "backend": "rawpy",
        "conditioning": "false",
    },
    "XFormerXTrans": {
        "url": "https://github.com/rymuelle/RawForge/releases/download/xtrans_v1.0.0/xformer.pt",
        "filename": "xformer.pt",
        "backend": "rawpy",
        "conditioning": "false",
        "batch_size": 1, 
    }, 
    "XFormerXTrans352": {
        "url": "https://github.com/rymuelle/RawForge/releases/download/xtrans_v1.0.0/xformer_352.pt",
        "filename": "xformer_352.pt",
        "backend": "rawpy",
        "conditioning": "false",
        "batch_size": 1, 
    }, 
    "RestormerXTrans": {
        "url": "https://github.com/rymuelle/RawForge/releases/download/xtrans_v1.0.0/restormer.pt",
        "filename": "restormer.pt",
        "backend": "rawpy",
        "conditioning": "false",
        "batch_size": 1, 
    }, 
}