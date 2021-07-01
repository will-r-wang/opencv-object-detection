# Computer Vision Task

## About this repo
**Introduction:** This repo contains the two part task for the Computer Vision technical interview portion of the transportation deep learning research team at Carleton.

|                |                                                                                                                                      |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Current status | Complete |
| In progress    | Follow https://github.com/will-r-wang/opencv-object-counting for what we're currently working on |
| Owner          | This repo is primarily maintained by individual contributors |
| Help           | For developer or design related questions, please send an email to william.wang@hey.com |

## How to use this repo
### Requirements
Here are some things you need to do in order to get started with opencv-object-counting:
- Follow the [OpenCV MacOS installation guide](https://learnopencv.com/install-opencv3-on-macos/) to get setup with opencv if you have not already done so
- Clone this repo and navigate to the root level with `git clone https://github.com/will-r-wang/opencv-object-counting`
- Install all the relevant dependencies with pip and download the YOLOv3-320 weights file into the model folder
  - First `pip3 install -r requirements.txt` for dependencies
  - Then go to https://pjreddie.com/darknet/yolo/ and download the YOLOv3-320 weights file and move it into `./models`
- Run `python3 yolo_video.py --input videos/input.mp4 --output output/output.avi` and you should be ready to go!

### Testing
The tasks in part B contain two basic base unit teststhat can be tested by running `python3 plot_histogram.py` and `python3 count_subset_occurrences.py` respectively.

### Instructions
See the file titled `instructions.pdf` for the full task instruction for this project.

### Technical Details
opencv-object-counting is released under the [MIT License](https://opensource.org/licenses/MIT).
