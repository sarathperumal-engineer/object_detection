from flask import Flask, render_template, request
import os
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageColor
import time
import base64

app = Flask(__name__)

def display_image(image):
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)

def download_and_resize_image(url, new_width=256, new_height=256, display=False):
    _, filename = tempfile.mkstemp(suffix=".jpg")
    response = urlopen(url)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.LANCZOS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    print("Image downloaded to %s." % filename)
    if display:
        display_image(pil_image)
    return filename

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

    display_str_heights = [font.getbbox(ds)[3] for ds in display_str_list]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height

    for display_str in display_str_list[::-1]:
        bbox = font.getbbox(display_str)
        text_width, text_height = bbox[2], bbox[3]
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
        text_bottom -= text_height - 2 * margin

def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image

@app.route('/')
def upload_file():
   return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join('uploads', f.filename)
        f.save(file_path)
        
        # Object detection code
        module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
        detector = hub.load(module_handle).signatures['default']

        def load_img(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            return img

        def run_detector(detector, path):
            img = load_img(path)

            converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
            start_time = time.time()
            result = detector(converted_img)
            end_time = time.time()

            result = {key: value.numpy() for key, value in result.items()}

            print("Found %d objects." % len(result["detection_scores"]))
            print("Inference time: ", end_time - start_time)

            image_with_boxes = draw_boxes(
                img.numpy(), result["detection_boxes"],
                result["detection_class_entities"], result["detection_scores"])

            # Convert the NumPy array to a PIL Image
            image_with_boxes_pil = Image.fromarray(image_with_boxes)

            # Save the image to a buffer (or a file)
            buffered = BytesIO()
            image_with_boxes_pil.save(buffered, format="JPEG")

            # Convert the image buffer to a base64-encoded string
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return encoded_image

        detected_image = run_detector(detector, file_path)
        return render_template('result.html', detected_image=detected_image)

if __name__ == '__main__':
    app.run(debug=True,port=8000)
