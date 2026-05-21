# denoiser

Fork of [rymuelle/RawForge](https://github.com/rymuelle/RawForge)

```bash
#install
./init.sh

#run (denoise all *.dng in input)
./denoise.sh

#run (denoise and convert all *.dng in input), requires imagemagick
convert=png ./denoise.sh

#run (extra args)
convert=tif ./denoise.sh --model TreeNetDenoiseHeavy
```

Models available:

+ TreeNetDenoise
+ TreeNetDenoiseLight
+ TreeNetDenoiseSuperLight
+ TreeNetDenoiseHeavy
+ Deblur
+ DeepSharpen
+ TreeNetDenoiseXTrans
+ XFormerXTrans
+ XFormerXTrans352
+ RestormerXTrans
