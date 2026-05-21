from PIL import Image

supported_extensions = ['JPG', 'JPEG', 'PNG', 'TIF', 'TIFF'] 

def convert_image(src_filename, dest_filename):
    src_ext = get_extension(src_filename)
    if src_ext not in supported_extensions:
        return False
    
    dest_ext = get_extension(dest_filename)
    if dest_ext not in supported_extensions:
        return False

    img = Image.open(src_filename)
    img.convert('RGB').save(dest_filename)

    return True

def get_extension(filename):
    return filename.split('.')[-1].upper()
