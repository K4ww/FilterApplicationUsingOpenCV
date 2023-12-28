# Face Filter Application

## Prerequisites

Make sure you have the following installed:

- Python 3.x
- OpenCV
- Tkinter

Install the required packages using:

```bash
pip install opencv-python
```

## Usage
Run the script using the following command:
```bash
python face_filter_app.py
```

A Tkinter window will appear with a menu of filter options.

Select a filter from the menu or choose "Launch All" to apply all filters simultaneously.

Press 'q' to exit the application.
# Applies a Sepia tone to the video feed.
python face_filter_app.py --filter sepia
# Adjusts the contrast of the video feed.
python face_filter_app.py --filter contrast
# Detects faces and overlays sunglasses on the eyes.
python face_filter_app.py --filter glasses
# Detects faces and overlays a mustache on the mouth.
python face_filter_app.py --filter mustache
# Generates interactive snowflakes on the video.
python face_filter_app.py --filter snowflake
# Changes the background to white while keeping detected faces.
python face_filter_app.py --filter white_background
# Applies all filters simultaneously.
python face_filter_app.py --filter all
# Cancels any applied filter and displays the original video feed.
python face_filter_app.py --filter cancel

