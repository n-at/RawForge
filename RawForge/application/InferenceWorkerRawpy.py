import onnxruntime as ort
import numpy as np
from blended_tiling_numpy import TilingModule
from RawForge.application.postprocessing import match_colors_linear
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004
from tqdm import tqdm
import rawpy

class InferenceWorkerRawpy():
    def __init__(self, model, model_params, rh, conditioning, dims, tile_size=512, tile_overlap=0.25, batch_size=2, disable_tqdm=False):
        super().__init__()
        self.model = model
        self.model_params = model_params
        self.rh = rh
        self.conditioning = conditioning
        self.dims = dims
        self.tile_size = tile_size
        if 'tile_size' in model_params:
            self.tile_size =  model_params['tile_size']
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        if 'batch_size' in model_params:
            self.batch_size = model_params['batch_size']
        self._is_cancelled = False
        self.disable_tqdm = disable_tqdm

    def cancel(self):
        self._is_cancelled = True
    def get_image(self,backend):
        if backend=='Malvar2004':
            image_RGB = self.rh.as_rgb(dims=self.dims, demosaicing_func=demosaicing_CFA_Bayer_Malvar2004, colorspace='lin_rec2020', clip=True)
        elif backend=='rawpy':
            image_RGB = self.rh.rawpy_object.postprocess(
                        user_wb=[1, 1, 1, 1],
                        output_color=rawpy.ColorSpace.raw,
                        demosaic_algorithm= rawpy.DemosaicAlgorithm(3),
                        no_auto_bright=True,
                        use_camera_wb=False,
                        use_auto_wb=False,
                        gamma=(1, 1),
                        user_flip=0,
                        output_bps=16,
                        no_auto_scale=True,
                    ) / self.rh.rawpy_object.white_level
            if self.dims is not None:
                image_RGB = image_RGB[self.dims[0]:self.dims[1], self.dims[2]:self.dims[3]]
            image_RGB = image_RGB.transpose(2, 0, 1)

        return np.expand_dims(image_RGB, axis=0).astype(np.float16)

    def _tile_process(self, model_params):
        # Prepare Data
        
        image_RGB = self.get_image(model_params['backend'])
        print(image_RGB.shape)

        full_size = [image_RGB.shape[2], image_RGB.shape[3]]
        tile_size = [self.tile_size, self.tile_size]
        overlap = [self.tile_overlap, self.tile_overlap]
        # Tiling Setup
        tiling_module_rgb = TilingModule(tile_size=[s for s in tile_size], tile_overlap=overlap, base_size=[s for s in full_size])

        tiles_rgb = tiling_module_rgb.split_into_tiles(image_RGB)
        print(tiles_rgb.shape)
        
        batches_rgb = [tiles_rgb[i : i + self.batch_size] 
                        for i in range(0, len(tiles_rgb), self.batch_size)]
        # Conditioning Setup
        cond_tensor = np.array([self.conditioning]).astype(np.float32)
        cond_tensor[:, 0] /= 6400.
        cond_tensor[:, 1] = 0.
        cond_tensor = cond_tensor[:, 0:1]
        cond_tensor = cond_tensor.astype(np.float16)
        if 'cond_scale' in model_params:
            print(model_params['cond_scale'], cond_tensor)
            cond_tensor *= cond_tensor * model_params['cond_scale']
        print(cond_tensor)

        processed_batches = []
        
        # Inference Loop
        for i, (batch_rgb) in tqdm(enumerate(batches_rgb), disable=self.disable_tqdm):
            if self._is_cancelled: return None, None
            
            B = batch_rgb.shape[0]
            # Expand conditioning to match batch size
            curr_cond = np.broadcast_to(cond_tensor, (B, cond_tensor.shape[-1])).astype(np.float16)

            payload = {
                "input": batch_rgb,
                "cond": curr_cond
            }
            # Filter based on inputs
            model_inputs = {i.name for i in self.model.get_inputs()}
            filtered_inputs = {k: v for k, v in payload.items() if k in model_inputs}
            output = self.model.run(["output"], filtered_inputs)
            
            processed_batches.append(output[0])
                    
        # Rebuild
        tiles_out = np.concat(processed_batches, axis=0)
        stitched = tiling_module_rgb.rebuild_with_masks(tiles_out)

        if "affine" in self.model_params:
            stitched, _, _ = match_colors_linear(stitched, image_RGB)

        stitched = stitched[0]
        print(stitched.shape)
        return image_RGB[0].transpose(1, 2, 0), stitched.transpose(1, 2, 0)
    
    def run(self, model_params):
        img, denoised_img = self._tile_process(model_params)

        # Post-process blending
        blend_alpha = self.conditioning[1] / 100
        final_denoised = (denoised_img * (1 - blend_alpha)) + (img * blend_alpha)
        
        return img, final_denoised
            