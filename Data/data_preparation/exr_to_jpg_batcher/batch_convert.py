from pathlib import Path
import sys, os
import imageio

input_dir  = Path(r"D:\Studium\ML43D\rerender_with_depth\output")
output_dir = Path(os.getcwd()) / "output/"

# source: https://stackoverflow.com/questions/50748084/convert-exr-to-jpeg-using-imageio-and-python
def convert_exr_to_jpg(exr_file, jpg_file):
    # if not os.path.isfile(exr_file):
    #     return False

    # filename, extension = os.path.splitext(exr_file)
    # if not extension.lower().endswith('.exr'):
    #     return False

    # imageio.plugins.freeimage.download() #DOWNLOAD IT
    image = imageio.imread(exr_file)
    print(image.dtype)

    # remove alpha channel for jpg conversion
    image = image[:,:,:3]


    data = 65535 * image
    data[data>65535]=65535
    rgb_image = data.astype('uint16')
    print(rgb_image.dtype)
    #rgb_image = imageio.core.image_as_uint(rgb_image, bitdepth=16)

    imageio.imwrite(jpg_file, rgb_image, format='jpeg')
    return True

i = 0

for obj_dir in input_dir.iterdir():
    dir_name = obj_dir.parts[-1]

    print(str(output_dir/dir_name))
    os.makedirs(os.path.dirname(str(output_dir/dir_name)+"/"))


    for file in (input_dir / obj_dir).iterdir():
        file_name = file.parts[-1]
        if file_name.startswith("rgb"):
            file_name = file_name.split(".")[0] + ".jpeg"
            convert_exr_to_jpg(str(file), str(output_dir/dir_name/file_name))
            print(file_name)
            i += 1

print(i)
