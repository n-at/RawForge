import os
from  RawForge.application.ModelHandler import ModelHandler 
from RawForge.application.postprocessing import postprocess
import RawForge.application.convert as convert
import argparse

def main():
    parser = argparse.ArgumentParser(description='A command line utility for processing raw images.')
    parser.add_argument('--model', type=str, help='The name of the model to use.', default='TreeNetDenoise')
    parser.add_argument('--in_file', type=str, help='The name of the file to open.')
    parser.add_argument('--out_file', type=str, help='The name of the file to save.')
    parser.add_argument('--conditioning', type=str, help='Conditioning array to feed model. Input string of numbers like so: 1,2,3')
    parser.add_argument('--dims', type=int, nargs=4, metavar=("x0", "x1", "y0", "y1"), help='Optional crop dimensions.')

    parser.add_argument('--cfa', action='store_true', help='Save the image as a CFA image (default: False).')
    parser.add_argument('--device', type=str, help='Set device backend (cuda, cpu, mps).')
    parser.add_argument('--disable_tqdm', action='store_true', help='Disable the progress bar.')
    parser.add_argument('--tile_size', type=int, help='Set tile size. (default: 256)', default=256)

    parser.add_argument('--lumi', type=float, help='Lumi noise (0-1).', default=0)
    parser.add_argument('--chroma', type=float, help='Chroma noise (0-1).', default=0)
    parser.add_argument('--clip_highlights', action='store_true', help='Do not run model on clipped highlights.')

    args = parser.parse_args()
    
    handler = ModelHandler()

    if not args.model:
        print('--model required')
        return

    handler.load_model(args.model)

    if not args.in_file:
        print('--in_file required')
        return

    if not args.out_file:
        print('--out_file required')
        return

    original_input_file = args.in_file
    original_output_file = args.out_file

    iso = handler.load_rh(original_input_file)

    if not args.conditioning:
        conditioning  = [iso, 0]
    else: 
        conditioning = [int(x) for x in args.conditioning.split(',')]

    if args.device:
        handler.set_device(args.device)

    inference_kwargs = {
        "disable_tqdm": args.disable_tqdm,
        "tile_size": args.tile_size,
    }
    img, denoised_image = handler.run_inference(conditioning=conditioning, dims=args.dims, inference_kwargs=inference_kwargs)
    output = postprocess(img, denoised_image, lumi_blend=args.lumi, chroma_blend=args.chroma, eps=1e-6, clip_highlights=args.clip_highlights)
    handler.handle_full_image(output, original_output_file, args.cfa)

if __name__ == '__main__':
    main()
