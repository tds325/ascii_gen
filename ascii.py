from __future__ import annotations
import numpy as np
import os
from math import floor, sqrt
from PIL import Image
from textwrap import dedent

ASPECT_LEEWAY = 0.9
DELIMITER=1
MAX_OUTPUT_HEIGHT = 50
MAX_OUTPUT_WIDTH = 160
SMALL_DEFAULT = 10
MED_DEFAULT = 20
LRG_DEFAULT = 36
XLRG_DEFAULT = 75
COLOR_MAX=255
TRANSFORM=1
PRE_COMPUTE=1

# compensate for characters being more tall than wide
H_MULTIPLIER = 3

class ImageData:
    def __init__(self, image, ascii_dict_len):
        self.image = image
        self.ascii_dict_len = ascii_dict_len
        self.pre_compute = {}

    def set_sub_image(self, sub_image)->ImageData:
        self.sub_image = sub_image
        return self

    def set_image(self, image)->ImageData:
        self.image = image
        return self

    def set_ascii_dict_len(self, ascii_dict_len)->ImageData:
        self.ascii_dict_len = ascii_dict_len
        return self

    def set_pre_compute(self, pre_compute)->ImageData:
        self.pre_compute = pre_compute
        return self

    def set_pre_compute_pair(self, kv_pair:tuple[str,object])->dict:
        if(len(kv_pair) != 2):
            print("Error setting key/value pair: tuple not of length 2")
        else:
            self.pre_compute[kv_pair[0]] = kv_pair[1]
        return self.pre_compute

    def get_pre_compute_by_key(self, key:str):
        return self.pre_compute[key]

def main():
    files = (os.listdir('images'))

    imglist = input_images(files)
    imgnum = select_image(imglist)

    output_to_ascii(imglist[imgnum])

def decide_classification(image_info:ImageData, 
                          data_transform, delimiter) -> int:
    num = data_transform(image_info)
    ascii_class = delimiter(num, image_info)
    return ascii_class

def determine_image_size(image, size:str = "large"):
    height_dict = { "small"  : SMALL_DEFAULT,
                    "medium" : MED_DEFAULT, 
                    "large"  : LRG_DEFAULT,
                    "xlarge" : XLRG_DEFAULT }

    width, height = image.size
    aspect_ratio = width / height
    
    if width > MAX_OUTPUT_HEIGHT or height > MAX_OUTPUT_WIDTH:
        height = height_dict[size]
        width = int(height * aspect_ratio * H_MULTIPLIER)

    aspect_ratio = width / height

    if(not validate_size(image.size[0], width)):
        width = (image.size[0] - 1) // 2
    if(not validate_size(image.size[1], height)):
        height = (image.size[1] - 1) // 2

    if(width / height < aspect_ratio):
        print("adjusting height to fit aspect ratio...")
        while((width / height < aspect_ratio) and aspect_ratio - width/height > ASPECT_LEEWAY ):
            height -= 1
    else:
        print("adjusting width to fit aspect ratio...")
        while(width / height > aspect_ratio and width/height - aspect_ratio > ASPECT_LEEWAY):
            width -= 1

    return (width,height)

def determine_segment_values(image, segment_list, dict_len):
    segment_key_list = []
    image_data = ImageData(image, dict_len)

    pre_compute(image_data)

    for segment in segment_list:
        sub_image = image[segment[2]:segment[3], segment[0]:segment[1]]
        if 0 in sub_image.shape:
            print(dedent(f'''\
            segment: {segment}
            sub_image: {sub_image}
            ERROR: segment of image with at least one axis of dimension 0.
            Check image segmentation logic - exiting program.'''))
            quit()
        image_data.set_sub_image(sub_image)
        segment_key_list.append(sub_image_to_key(image_data))
    return segment_key_list

def input_images(files):
    imglist = []
    for img_name in files:
        try:
            image = Image.open('images/'+img_name)
            image.name = img_name
            imglist.append(image)
        except:
            print(f'unable to open file {img_name}')
    return imglist

def n_exclusive_segments(axis_size:int, num_segments:int)->list:
    seg_list = []

    if (num_segments) > (axis_size + 1)/2:
        print(dedent(f'''\
            Number of segments too large for axis with too few pixels.  
            ({axis_size}, {num_segments}) Exiting program.'''))
        quit()

    pixel_width:int = 1
    temp:int = 1
    while((temp*num_segments)+(num_segments - 1) <= axis_size):
        pixel_width = temp
        temp += 1

    # calculate 'leftovers' 
    # *remainder will always be less than the number of segments
    # otherwise the pixel_width could have simply been incremented by one
    leftover = axis_size - ((pixel_width * num_segments) + (num_segments - 1))

    # evenly distribute the remainder throughout the segments
    remainder_list = []
    middle_index = (num_segments // 2)
    remainder_list = [middle_index for i in range(leftover)]

    # subtract the distance from the center of the array
    for i in range(leftover):
        remainder_list[i] = remainder_list[i] + (i - leftover // 2)

    previous = -1
    for i in range(num_segments):
        endsegment_index:int = previous + pixel_width + 1
        if i in remainder_list:
            endsegment_index += 1
        temple:tuple = (previous+1, endsegment_index)
        seg_list.append(temple)
        previous = endsegment_index
        
    return seg_list

def output_to_ascii(image):
    ascii_dict = { 0: '%',
                   1: '#', 
                   2: '@',
                   3: '+',
                   4: '=', 
                   5: '-',
                   6: ';',
                   7: ':',
                   8: '.',
                   9: ' '
                  }

    width, height = determine_image_size(image)
    segment_list = scan_image(image, width, height)
    segment_list = validate_list(segment_list, image.size[0], image.size[1])
    np_arr = np.asarray(image)
    if(image.mode == "P"):
         global PRE_COMPUTE
         PRE_COMPUTE = 0
    segment_key_list = determine_segment_values(np.asarray(image), segment_list, len(ascii_dict))

    # Palette mode images store color values differently
    flip_key = lambda key, dict_len: (-key + (dict_len-1)) if image.mode == "P" or image.size[0]>275 else key

    index = 0
    while index < width * height:
        key = segment_key_list[index]
        key = flip_key(key, len(ascii_dict))
        print(ascii_dict[key], end='')
        if((index+1) % width == 0):
            print()
        index += 1

def pre_compute(image_data:ImageData)->ImageData:
    # check if need to pre-compute any data
    # default std-dev of r,g,b
    match(PRE_COMPUTE):
        case 1:
            std_dev = [np.std(image_data.image[:,:,i].flatten()) for i in range(3)]
            image_data.set_pre_compute_pair(("std-dev",std_dev))
        case 0:
            std_dev = np.std(image_data.image)

    total_mean = np.mean(image_data.image)

    distance_to_ends = (COLOR_MAX - total_mean, total_mean)

    max_n_root = [sqrt(dist) for dist in distance_to_ends]
    # allow for left and right ends to have some space
    # todo more accurate spacing
    max_n_root[0] *= 0.9
    max_n_root[1] *= 0.9

    # greater side gets more delimiting boundaries
    total_boundaries = image_data.ascii_dict_len - 1
    if(distance_to_ends[0] > distance_to_ends[1]):
        rhs = total_boundaries // 2
        lhs = total_boundaries - rhs
    else:
        lhs = total_boundaries // 2
        rhs = total_boundaries - lhs

    left_quotient = max_n_root[0] / lhs
    right_quotient = max_n_root[1] / rhs

    delimit_list = []
    for i in range(lhs,0,-1):
        delimit_list.append(total_mean + (left_quotient*i)**2)
    for i in range(1,rhs+1):
            delimit_list.append(total_mean - (right_quotient*i)**2)

    image_data.set_pre_compute_pair(("delimit_list",delimit_list))

    print(std_dev)
    print(delimit_list)
    return image_data

def print_imglist(imglist:list):
    for num, img in enumerate(imglist):
        print(f'{num}:\t{img.name}\t{img.size}\t{img.mode}')

def scan_image(image, width:int, height:int) -> list:
    image_width, image_height = image.size
    x_segments = n_exclusive_segments(image_width, width)
    y_segments = n_exclusive_segments(image_height, height)
    segment_list = []

    for y_seg in y_segments:
        for x_seg in x_segments:
            segment_list.append((x_seg[0], x_seg[1], y_seg[0], y_seg[1]))

    return segment_list

def select_image(imglist:list)->int:
    prompt = 'Please select an image to output in ascii: ' 
    print_imglist(imglist)

    imgnum = tryparse_int(input(prompt))
    while(imgnum not in range(len(imglist))):
        print(f'image #{imgnum} is not a valid selection')
        imgnum = tryparse_int(input(prompt))
    return imgnum

def sub_image_to_key(image_data:ImageData)->int:
    default_transform = lambda image_data : np.mean(image_data.sub_image)
    default_delimiter = lambda num, image_data: floor(num / (COLOR_MAX / (image_data.ascii_dict_len - 1)))

    match TRANSFORM:
        case 0: 
            transform = default_transform
        case 1:
            transform = lambda image_data: np.median(image_data.sub_image)
        case _:
            transform = default_transform

    match DELIMITER:
        case 0:
            delimit = default_delimiter
        case 1: 
            # group delimiting numbers around mean
            def squared_dist_from_mean_delimit(num, image_data):
                ascii_key = image_data.ascii_dict_len - 1
                for delimiter in image_data.get_pre_compute_by_key("delimit_list"):
                    if(num < delimiter):
                        ascii_key -= 1
                
                return ascii_key
            delimit=squared_dist_from_mean_delimit
        case _:
            delimit = default_delimiter

    return decide_classification(image_data, transform, delimit)

def tryparse_int(str : str) -> int:
    val = -1
    try:
        val = int(str)
    except:
        print('Not a valid integer. Exiting Program.')
        quit()
    return val

def validate_list(segment_list:list, width:int, height:int) -> list:
    for i in range(0, len(segment_list)):
        temp = list(segment_list[i])
        if(temp[0] > width):
            temp[0] = width
        if(temp[1] > width):
            temp[1] = width
        if(temp[2] > height):
            temp[2] = height
        if(temp[3] > height):
            temp[3] = height
        segment_list[i] = tuple(temp)
    return segment_list

def validate_size(axis_size:int, num_segments:int)->bool:
    return not (num_segments) > (axis_size + 1)/2

if __name__ == '__main__':
    main()
