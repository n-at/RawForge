from PIL import Image
import piexif
import tifffile

def get_from_jpg(file_name):
    exif_dict = {}

    img = Image.open(file_name)
    
    raw_exif = img.info.get('exif', b'')
    if not raw_exif:
        return {}
    
    loaded_exif = piexif.load(raw_exif)

    for tag_id, data in loaded_exif['Exif'].items():
        property = piexif.TAGS['Exif'].get(tag_id, None)
        if property is None:
            continue
        exif_dict[property['name']] = data
        
    return exif_dict

def get_from_png(file_name):
    return {}

def get_from_tiff(file_name):
    image = tifffile.TiffFile(file_name)
    tags = {}
    for tag in image.pages[0].tags:
        if tag.code == 34665:
            tags.update(tag.value)
    return tags
