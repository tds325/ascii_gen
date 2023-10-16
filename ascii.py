import numpy as np
import os
from math import floor, ceil
from PIL import Image
from textwrap import dedent

MAX_OUTPUT_HEIGHT = 50
MAX_OUTPUT_WIDTH = 160
SMALL_DEFAULT = 10
MED_DEFAULT = 20
LRG_DEFAULT = 36
XLRG_DEFAULT = 75
COLOR_MAX=255
TRANSFORM=1
DELIMITER=0

# compensate for characters being more tall than wide
H_MULTIPLIER = 3

def main():
    files = (os.listdir('images'))

    imglist = input_images(files)
    imgnum = select_image(imglist)

    output_to_ascii(imglist[imgnum])

def decide_classification(sub_image, image, dict_len, 
                          data_transform, delimiter) -> int:
    num = data_transform(sub_image, image)
    ascii_class = delimiter(num, dict_len)
    return ascii_class

def determine_image_size(image, size:str = "xlarge"):
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
        while((width / height < aspect_ratio) and aspect_ratio - width/height > 0.5):
            height -= 1
    else:
        print("adjusting width to fit aspect ratio...")
        while(width / height > aspect_ratio and width/height - aspect_ratio > 0.5):
            width -= 1

    return (width,height)

def determine_segment_values(image, segment_list, dict_len):
    segment_key_list = []
    for segment in segment_list:
        sub_image = image[segment[2]:segment[3], segment[0]:segment[1]]
        if 0 in sub_image.shape:
            print(dedent(f'''\
            segment: {segment}
            sub_image: {sub_image}
            ERROR: segment of image with at least one axis of dimension 0.
            Check image segmentation logic - exiting program.'''))
            quit()
        segment_key_list.append(sub_image_to_key(sub_image, dict_len, image))
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
    #print(f"pixel_width: {pixel_width}, ({axis_size}, {num_segments}), r {leftover}")

    # evenly distribute the remainder throughout the segments
    #partition_quotient = num_segments / leftover
    remainder_list = []
    middle_index = (num_segments // 2)
    remainder_list = [middle_index for i in range(leftover)]

    # precipitate from center by adding/subtracting from segment index
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
    segment_key_list = determine_segment_values(np.asarray(image), segment_list, len(ascii_dict))

    index = 0
    while index < width * height:
        print(ascii_dict[segment_key_list[index]], end='')
        if((index+1) % width == 0):
            print()
        index += 1

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

#def scan_image(image, width:int, height:int) -> list:
#    image_width, image_height = image.size
#    pixel_width = image_width / width
#    pixel_height = image_height / height
#
#    segment_list = []
#
#    for h_index in range(height):
#        for w_index in range(width):
#            start_width = (w_index * pixel_width)
#            start_height = (h_index * pixel_height)
#            segment_list.append((round(start_width), 
#                                 round(start_width+pixel_width), 
#                                 round(start_height), 
#                                 round(start_height+pixel_height)))
#            # no overlapping pixels
#            temp = list(segment_list.pop())
#            if(h_index != height -1 and w_index != width - 1):
#                temp[1] = temp[1] - 1
#                temp[3] = temp[3] - 1
#            elif(h_index != height - 1):
#                temp[3] = temp[3] - 1
#            elif(w_index != width - 1):
#                temp[1] = temp[1] - 1
#            segment_list.append(tuple(temp))
#    return segment_list

def select_image(imglist:list)->int:
    prompt = 'Please select an image to output in ascii: ' 
    print_imglist(imglist)

    imgnum = tryparse_int(input(prompt))
    while(imgnum not in range(len(imglist))):
        print(f'image #{imgnum} is not a valid selection')
        imgnum = tryparse_int(input(prompt))
    return imgnum

def sub_image_to_key(sub_image, dict_len:int, image)->int:
    default_transform = lambda sub_image, image: np.mean(sub_image)
    default_delimiter = lambda num, dict_len: floor(num / (COLOR_MAX / (dict_len - 1)))

    match TRANSFORM:
        case 0: 
            transform = default_transform
        case 1:
            transform = lambda sub_image, image: np.median(sub_image)
        case _:
            transform = default_transform

    match DELIMITER:
        case 0:
            delimit = default_delimiter
        case _:
            delimit = default_delimiter

    return decide_classification(sub_image, image, dict_len, transform, delimit)

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
