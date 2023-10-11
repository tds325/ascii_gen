import numpy as np
import os
from math import floor
from PIL import Image

MAX_OUTPUT_HEIGHT = 50
MAX_OUTPUT_WIDTH = 160
SMALL_DEFAULT = 10
MED_DEFAULT = 20
LRG_DEFAULT = 36

# compensate for characters being more tall than wide
H_MULTIPLIER = 3

def main():
    files = (os.listdir('images'))

    imglist = input_images(files)
    imgnum = select_image(imglist)

    output_to_ascii(imglist[imgnum])

def determine_image_size(img, size = "large"):
    height_dict = { "small" : SMALL_DEFAULT,
                    "medium" : MED_DEFAULT, 
                    "large" : LRG_DEFAULT }

    width, height = img.size
    aspect_ratio = width / height
    
    if width > MAX_OUTPUT_HEIGHT or height > MAX_OUTPUT_WIDTH:
        height = height_dict[size]
        width = int(height * aspect_ratio * H_MULTIPLIER)
        
    return (width,height)

def determine_segment_values(image, segment_list, dict_len):
    segment_key_list = []
    for segment in segment_list:
        sub_image = image[segment[2]:segment[3], segment[0]:segment[1]]
        segment_key_list.append(sub_image_to_key(sub_image, dict_len))
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

def print_imglist(imglist):
    for num, img in enumerate(imglist):
        print(f'{num}:\t{img.name}\t{img.size}\t{img.mode}')

def scan_image(image, width, height):
    image_width, image_height = image.size
    pixel_width = image_width / width
    pixel_height = image_height / height

    segment_list = []

    for h_index in range(height):
        for w_index in range(width):
            start_width = (w_index * pixel_width)
            start_height = (h_index * pixel_height)
            segment_list.append((round(start_width), 
                                 round(start_width+pixel_width), 
                                 round(start_height), 
                                 round(start_height+pixel_height)))
            # no overlapping pixels
            temp = list(segment_list.pop())
            if(h_index != height -1 and w_index != width - 1):
                temp[1] = temp[1] - 1
                temp[3] = temp[3] - 1
            elif(h_index != height - 1):
                temp[3] = temp[3] - 1
            elif(w_index != width - 1):
                temp[1] = temp[1] - 1
            segment_list.append(tuple(temp))
    return segment_list

def select_image(imglist):
    prompt = 'Please select an image to output in ascii: ' 
    print_imglist(imglist)

    imgnum = tryparse_int(input(prompt))
    while(imgnum not in range(len(imglist))):
        print(f'image #{imgnum} is not a valid selection')
        imgnum = tryparse_int(input(prompt))
    return imgnum

def sub_image_to_key(sub_image, dict_len):
    delimiter = 255 / (dict_len - 1)
    avg = np.average(sub_image)
    return floor(avg/delimiter) 

def tryparse_int(str):
    val = -1
    try:
        val = int(str)
    except:
        print('Not a valid integer. Exiting Program.')
        quit()
    return val

def validate_list(segment_list, width, height):
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

if __name__ == '__main__':
    main()
