import argparse
import RawForge.application.convert as convert
import RawForge.application.exif as exif

def main():
    parser = argparse.ArgumentParser(description='EXIF tags extractor')
    parser.add_argument('--in_file', type=str, help='The name of the file to open.')
    args = parser.parse_args()

    if not args.in_file:
        print('--in_file required')
        return

    ext = convert.get_extension(args.in_file)

    if ext == 'JPG' or ext == 'JPEG':
        tags = exif.get_from_jpg(args.in_file)
    elif ext == 'PNG':
        tags = exif.get_from_png(args.in_file)
    elif ext == 'TIF' or ext == 'TIFF' or ext == 'DNG':
        tags = exif.get_from_tiff(args.in_file)
    else:
        print("Unsupported file type")

    print(tags)

if __name__ == '__main__':
    main()
