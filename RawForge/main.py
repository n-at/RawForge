import sys
import platform
import argparse
from pathlib import Path
from  RawForge.application.ModelHandler import ModelHandler 
from RawForge.application.postprocessing import postprocess
from  RawForge.application.MODEL_REGISTRY import MODEL_REGISTRY 
from RawForge.application.InferenceWorker import InferenceWorker
from RawForge.application.ImageSaver import ImageSaver
from RawForge.application.helpers.get_image import get_image
import numpy as np

# import glob

def main():
    parser = argparse.ArgumentParser(description='A command line utility for processing raw images.')
    parser.add_argument('model', type=str, help='The name of the model to use.')
    parser.add_argument('in_file', type=str, help='The name of the file to open.')
    parser.add_argument('out_file', type=str, help='The name of the file to save.')
    parser.add_argument('--conditioning', type=str, help='Conditioning array to feed model. Input string of numbers like so: 1,2,3')
    parser.add_argument('--dims', type=int, nargs=4, metavar=("x0", "x1", "y0", "y1"), help='Optional crop dimensions.')

    parser.add_argument('--cfa', action='store_true', help='Save the image as a CFA image (default: False).')
    parser.add_argument('--device', type=str, help='Set device backend (cuda, cpu, mps).')
    parser.add_argument('--disable_tqdm', action='store_true', help='Disable the progress bar.')
    parser.add_argument('--tile_size', type=int, help='Set tile size. (default: 256)', default=256)
    parser.add_argument('--tile_overlap', type=float, help='Set tile overlap. (default: 0.25)', default=0.25)

    parser.add_argument('--lumi', type=float, help='Lumi noise (0-1).', default=0)
    parser.add_argument('--chroma', type=float, help='Chroma noise (0-1).', default=0)
    parser.add_argument('--clip_highlights', action='store_true', help='Do not run model on clipped highlights.')
    parser.add_argument('--affine', action='store_true', help='Affine fit the model to the input.')
    # # Glob handeling
    # in_files = sorted(glob.glob(args.in_file))
    # if not in_files:
    #     raise FileNotFoundError(f"No files match pattern: {args.in_file}")
    # if len(in_files) > 1:
    #     if not args.out_path.is_dir():
    #         raise ValueError(
    #             "When using glob input, out_file must be a directory."
    #         )
    args = parser.parse_args()

    # Initialize
    models = args.model.split(',')
    model_params = MODEL_REGISTRY[models[0]]
    rh, image_RGB, iso = get_image(args.in_file, model_params, dims=args.dims)
    if not args.conditioning:
        conditioning  = [iso, 0]
    else: 
        conditioning = [int(x) for x in args.conditioning.split(',')]
    inference_kwargs = {"disable_tqdm": args.disable_tqdm,
                        "tile_size": args.tile_size,
                        "tile_overlap": args.tile_overlap}
    
    # Run processing
    output_img = image_RGB
    for model in models:
        handler = ModelHandler()
        handler.load_model(model)
        worker = InferenceWorker(handler.model, handler.model_params, conditioning, **inference_kwargs)
        _, output_img = worker._tile_process(output_img, handler.model_params)

    # Postprocess
    output = postprocess(image_RGB, output_img, lumi_blend=args.lumi, chroma_blend=args.chroma, eps=1e-6,
                         clip_highlights=args.clip_highlights, affine=args.affine)
    
    # Save image
    saver = ImageSaver(model_params, rh, dims=args.dims)
    apply_ccm = model_params['demosaicing'] == 'rawpy'
    if Path(args.out_file).suffix=='.tiff':
        saver.to_tiff(output, args.out_file, apply_ccm=apply_ccm)
    else:
        saver.to_raw(output, args.out_file, args.cfa)

if __name__ == '__main__':
    main()
