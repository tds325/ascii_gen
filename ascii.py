from __future__ import annotations
import argparse
import typing
import time
from functools import reduce
import numpy as np
import os
from math import floor, ceil, sqrt
from PIL import Image
from textwrap import dedent

ASPECT_LEEWAY = 0.9
COLOR_MAX=255
DELIMITER=0
H_MULTIPLIER = 3
MAX_OUTPUT_HEIGHT = 50
MAX_OUTPUT_WIDTH = 160
PRE_COMPUTE=0
TRANSFORM=1

SMALL_DEFAULT = 20
MED_DEFAULT = 36
LRG_DEFAULT = 65
XLRG_DEFAULT = 88

class ImageData:
    def __init__(self, image, ascii_dict_len):
        self.image = image
        self.ascii_dict_len = ascii_dict_len
        self.pre_compute = {}

    def set_sub_image(self, sub_image:np.ndarray)->ImageData:
        self.sub_image = sub_image
        return self

    def set_image(self, image:np.ndarray)->ImageData:
        self.image = image
        return self

    def set_ascii_dict_len(self, ascii_dict_len:int)->ImageData:
        self.ascii_dict_len = ascii_dict_len
        return self

    def set_pre_compute(self, pre_compute:dict[str,typing.Any])->ImageData:
        self.pre_compute = pre_compute
        return self

    def set_pre_compute_pair(self, kv_pair:tuple[str,typing.Any])->dict[str,object]:
        if(len(kv_pair) != 2):
            print("Error setting key/value pair: tuple not of length 2")
        else:
            self.pre_compute[kv_pair[0]] = kv_pair[1]
        return self.pre_compute

    def get_pre_compute_by_key(self, key:str):
        return self.pre_compute[key]

def main():
    parser = initialize_arg_parser()
    args = parser.parse_args()

    initialize_globals(args)

    image = Image.Image()
    if(args.file != None):
        image = open_image(args.file)
    else:
        files = (os.listdir(args.directory))
        imglist = input_images(files, args.directory)
        imgnum = select_image(imglist)
        image = imglist[imgnum]

    output_to_ascii(image, args)

def adjust_to_aspect_ratio(aspect_ratio:float, width:int, height:int)->tuple[int,int]:
    if(width / height < aspect_ratio):
        while((width / height < aspect_ratio) and aspect_ratio - width/height > ASPECT_LEEWAY ):
            height -= 1
    else:
        while(width / height > aspect_ratio and width/height - aspect_ratio > ASPECT_LEEWAY):
            width -= 1
    return (width,height)
    
def decide_classification(image_info:ImageData, 
                          data_transform: typing.Callable[[ImageData], float], 
                          delimiter: typing.Callable[[float,ImageData],int]) -> int:
    num = data_transform(image_info)
    ascii_class = delimiter(num, image_info)
    return ascii_class

def determine_image_size(image:Image.Image, size:str)->tuple[int, int]:
    height_dict = { "small"  : SMALL_DEFAULT,
                    "medium" : MED_DEFAULT, 
                    "large"  : LRG_DEFAULT,
                    "xlarge" : XLRG_DEFAULT }

    width, height = image.size
    aspect_ratio = width / height
    
    height = height_dict[size]
    width = int(height * aspect_ratio * H_MULTIPLIER)

    aspect_ratio = width / height

    if(not validate_size(image.size[0], width)):
        width = (image.size[0] - 1) // 2
    if(not validate_size(image.size[1], height)):
        height = (image.size[1] - 1) // 2

    width, height = adjust_to_aspect_ratio(aspect_ratio, width, height)

    return (width,height)

def determine_segment_values(image_data:ImageData, 
                             segment_list:list[tuple[int,int,int,int]])->list[int]:
    segment_key_list = []

    for segment in segment_list:
        sub_image = image_data.image[segment[2]:segment[3], segment[0]:segment[1]]
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

def gamma_expansion(np_arr:np.ndarray)->np.ndarray:
    np_arr = np.where(np_arr <= 0.04045, np_arr / 12.92, ((np_arr+0.055)/1.055)**2.4)
    return np_arr

def histogram_from_arr(np_arr:np.ndarray, size:int)->np.ndarray:
    histogram = np.asarray([0 for _ in range(size)])
    if(len(np_arr.shape)==3):
        temp_arr = np_arr[:,:,0]
        row, col, dim = np_arr.shape
    else:
        temp_arr = np_arr
        row, col = np_arr.shape
    for r in range(row):
        for c in range(col):
            temp_arr[r,c] = np.mean(np_arr[r,c])
            histogram[tryparse_int(temp_arr[r,c]*size-1)] += 1
    histogram = histogram / (row*col)
    return histogram

def histogram_normalization(np_arr:np.ndarray)->np.ndarray:
    histogram = histogram_from_arr(np_arr, COLOR_MAX+1)

    # find first and last nonzero
    nonzero = [i for i,bool in enumerate(np.where(histogram != 0, True, False)) if bool]
    if(len(nonzero) > 0):
        first, last = nonzero[0], nonzero[-1]

        nonzero = nonzero[first:last+1]

        cdf = [reduce(lambda x,y: x+y, histogram[:i]) for i in range(1,COLOR_MAX+2)]
        #cdf = [np.ceil(x * COLOR_MAX) for x in cdf]

        # histogram normalization algorithm
        stretch = lambda x: np.asarray([cdf[round(y*COLOR_MAX)] for y in x])

        row,col = np_arr.shape[:2]
        for r in range(row):
            for c in range(col):
                np_arr[r,c,:] = stretch(np_arr[r,c,:])

    return np_arr

def initialize_arg_parser()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Create ascii art from images')
    parser.add_argument('-n', '--n-chars', 
                        help="Number of possible characters to be derived from the image",
                        default=71,
                        required = False)
    parser.add_argument('-f', '--file', 
                        help="Choose an image file directly instead of through the program", 
                        required = False)
    parser.add_argument('-d', '--directory', 
                        help="Choose an image directory to select a possible image from",
                        default='images/',
                        required = False)
    parser.add_argument('-s', '--size', 
                        help="image size selection from [small, medium, large, xlarge]",
                        default = "large",
                        required = False)
    parser.add_argument('-o', '--output',
                        help="output to a file instead of stdio",
                        required = False)
    parser.add_argument('-dl', '--delimiter',
                        help="define how image classifications are spread among possible values",
                        default = 1,
                        required = False)
    parser.add_argument('-bg', '--background',
                        help="output intended for light or dark background",
                        default = "dark",
                        required = False)
    #parser.add_argument('-r', '--range',
    #                    help="select a range more narrow (<255) in cases of images where the values are too similar in regular use",
    #                    required = False)

    return parser

def initialize_ascii_dict(size:int)->dict[int,str]:
    selection_arr = [ 1, 3, 8, 51, 52, 54, 62, 63, 69, 70 ] 
    min_size = len(selection_arr)

    ascii_dict = { 0: '$', 1: '@', 2: 'B', 3: '%', 4: '8',
                   5: '&', 6: 'W', 7: 'M', 8: '#', 9: '*', 
                   10: 'o', 11: 'a', 12: 'h', 13: 'k', 14: 'b', 
                   15: 'd', 16: 'p', 17: 'q', 18: 'w', 19: 'm', 
                   20: 'Z', 21: 'O', 22: '0', 23: 'Q', 24: 'L', 
                   25: 'C', 26: 'J', 27: 'U', 28: 'Y', 29: 'X', 
                   30: 'z', 31: 'c', 32: 'v', 33: 'u', 34: 'n', 
                   35: 'x', 36: 'r', 37: 'j', 38: 'f', 39: 't', 
                   40: '/', 41: '\\', 42: '|', 43: '(', 44: ')', 
                   45: '1', 46: '{', 47: '}', 48: '[', 49: ']', 
                   50: '?', 51: '=', 52: '-', 53: '_', 54: '+', 
                   55: '~', 56: '<', 57: '>', 58: 'i', 59: '!', 
                   60: 'l', 61: 'I', 62: ';', 63: ':', 64: ',', 
                   65: '"', 66: '^', 67: '`', 68: '\'', 69: '.', 
                   70: ' ' }

    if(size < min_size or size > len(ascii_dict)):
        print("Invalid dictionary size, exiting program.")
        quit()
    elif(size != len(ascii_dict)):
        def add_to_selection_arr(s_arr:list[int])->list[int]:
            diff = []
            s_arr.insert(0,-1)
            for x, y in zip(s_arr[0:], s_arr[1:]):
                diff.append(y-x)

            max = diff[0]
            max_index = 0
            for i, num in enumerate(diff):
                if num > max:
                    max = num
                    max_index = i
            if max < 2:
                print("Unable to expand array further, exiting program")
                quit()
            s_arr.insert(max_index+1, (s_arr[max_index]+s_arr[max_index+1])//2)
            s_arr.pop(0)
            return s_arr

        # interpolate new values to dict selection array
        for _ in range(size-min_size):
            selection_arr = add_to_selection_arr(selection_arr)

        new_dict = {}
        for selection in selection_arr:
            new_dict[selection] = ascii_dict[selection]
        ascii_dict = {}

        # reset key values to be 0 -> len(dict) - 1
        for i, value in enumerate(new_dict.values()):
            ascii_dict[i] = value

    return ascii_dict

def initialize_globals(args:argparse.Namespace):
    if(args.delimiter != None):
        global DELIMITER
        DELIMITER = tryparse_int(args.delimiter)

def input_images(files:list, dir:str)->list[Image.Image]:
    imglist = []
    for img_name in files:
        image = open_image(dir+img_name)
        image.name = img_name
        imglist.append(image)
    return imglist

def n_exclusive_segments(axis_size:int, num_segments:int)->list[tuple[int,int]]:
    seg_list = []

    if (not validate_size(axis_size, num_segments)):
        print(dedent(f'''\
            Number of segments too large for axis with too few pixels.  
            ({axis_size}, {num_segments}) Exiting program.'''))
        quit()

    pixel_width = 1
    temp = 1
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
    remainder_list = [middle_index for _ in range(leftover)]

    # subtract the distance from the center of the array
    for i in range(leftover):
        remainder_list[i] = remainder_list[i] + (i - leftover // 2)

    previous = -1
    for i in range(num_segments):
        endsegment_index = previous + pixel_width + 1
        if i in remainder_list:
            endsegment_index += 1
        temple = (previous+1, endsegment_index)
        seg_list.append(temple)
        previous = endsegment_index
        
    return seg_list

def open_image(file:str)->Image.Image:
    image = Image.Image()
    try:
        image = Image.open(file)
    except OSError:
        print(f'unable to open file {str}')
    image = image if(image.mode == "RGB") else Image.Image.convert(image, mode="RGB")
    return image

def output_to_ascii(image:Image.Image, args:argparse.Namespace):
    ascii_dict = initialize_ascii_dict(tryparse_int(args.n_chars))
    np_arr:np.ndarray = preprocess_image(np.asarray(image))
    image_data = ImageData(np_arr, len(ascii_dict))

    width, height = determine_image_size(image, args.size)
    segment_list = segment_image(image, width, height)

    pre_compute(image_data)
    segment_key_list = determine_segment_values(image_data, segment_list)

    # Palette mode images store color values differently
    std_dev = np.mean(image_data.get_pre_compute_by_key("std-dev"))

    flip_key = lambda key, dict_len: (-key + (dict_len-1)) if args.background == "dark" else key

    output = ""
    index = 0
    while index < width * height:
        key = segment_key_list[index]
        key = flip_key(key, len(ascii_dict))
        output+=ascii_dict[key]
        if((index+1) % width == 0):
            output+="\n"
        index += 1

    if(args.output != None):
        try:
            args.output = os.path.join(os.getcwd(), args.output)
            file = open(args.output, 'w')
            file.write(output)
            file.close
        except OSError:
            print(f"Error writing output to file {args.output}, exiting program")
            quit()
    else:
        print(output)


def pre_compute(image_data:ImageData)->ImageData:
    match(PRE_COMPUTE):
        case _:
            std_dev = [np.std(image_data.image[:,:,i].flatten()) for i in range(3)]
    image_data.set_pre_compute_pair(("std-dev",std_dev))

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

    return image_data

def preprocess_image(image:np.ndarray)->np.ndarray:
    start = time.time()
    
    temp_arr = image / COLOR_MAX
    
    temp_arr = gamma_expansion(temp_arr)
    #linear_approx = lambda rgb: [0.299*rgb[0], 0.587*rgb[1], 0.144*rgb[2]]

    # scalar value between 0 and 1 for interpolation of normalized image 
    # to increase clarity of image, set above 1.0 (some information will be lost)
    # TODO: how to decide best scalar? *add default strategy and cmd line option
    scalar = 0.0
    hist_arr = histogram_normalization(temp_arr.copy())
    temp_arr = (temp_arr*(1.0-scalar) + hist_arr*scalar)
    temp_arr = np.clip(temp_arr,0.0,1.0)

    # TODO: add dithering to compensate for quantization error

    image = temp_arr * COLOR_MAX
    print(time.time() - start)
    return image

def segment_image(image:Image.Image, width:int, height:int) -> list[tuple[int,int,int,int]]:
    image_width, image_height = image.size
    x_segments = n_exclusive_segments(image_width, width)
    y_segments = n_exclusive_segments(image_height, height)
    segment_list = []

    for y_seg in y_segments:
        for x_seg in x_segments:
            segment_list.append((x_seg[0], x_seg[1], y_seg[0], y_seg[1]))

    return segment_list

def select_image(imglist:list[Image.Image])->int:
    prompt = 'Please select an image to output in ascii: ' 

    for num, img in enumerate(imglist):
        print(f'{num}:\t{img.name}\t{img.size}\t{img.mode}')

    imgnum = tryparse_int(input(prompt))
    while(imgnum not in range(len(imglist))):
        print(f'image #{imgnum} is not a valid selection')
        imgnum = tryparse_int(input(prompt))
    return imgnum

def sub_image_to_key(image_data:ImageData)->int:
    default_transform = lambda image_data: np.mean(image_data.sub_image)
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
    except ValueError:
        print('Not a valid integer. Exiting Program.')
        quit()
    return val

def validate_size(axis_size:int, num_segments:int)->bool:
    return not (num_segments) > (axis_size + 1)/2

if __name__ == '__main__':
    main()
