# AsciiGen

Convert images to ascii style using python.

`$ python3 ascii.py` defaults to looking within the local dir for an *images/* directory, which then numbers all images allowing you to select interactively which image to convert. The `-d` or `--directory` option will let you choose an alternative, such as ~/Pictures.

`$ python3 ascii.py --file image.<type>` runs the program non-interactively and is generally faster.

### Arguments
---
| Short flag | Long flag | Description |
| --- | --- | :--- |
| `-n`  | `--n-chars`     | number of possible characters |
| `-f`  | `--file`        | choose an image file directly instead of through the program |
| `-d`  | `--directory`   | specify the image directory to scan when starting the program |
| `-s`  | `--size`        | output size selection: [**small**, **medium**, **large**, **xlarge**] |
| `-o`  | `--output`      | output directly to file instead of stdio |
| `-dl` | `--delimiter`   | define strategy in which characters are selected from image values |
| `-bg` | `--background`  | output intended for light or dark background (default is dark) |


![Gif of running program](https://github.com/tds325/ascii_gen/blob/main/frog.gif?raw=true)
