from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf
import tensorflow_hub as hub
from skimage.measure import ransac
from skimage.feature import plot_matches
from scipy.spatial import cKDTree
import base64
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform

app = Flask(__name__)

# Load DELF model
delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']

@app.route('/')
def index():
    return render_template('feature.html')

@app.route('/upload', methods=['POST'])
def upload():
    image1_file = request.files['image1']
    image2_file = request.files['image2']

    # Read and preprocess images
    image1 = Image.open(image1_file)
    image2 = Image.open(image2_file)
    image1 = image1.convert('RGB')
    image2 = image2.convert('RGB')

    # Run DELF on images
    result1 = run_delf(image1)
    result2 = run_delf(image2)

    # Perform matching and get correspondences
    correspondences = match_images(image1, image2, result1, result2)

    return correspondences

def run_delf(image):
    np_image = np.array(image)
    float_image = tf.image.convert_image_dtype(np_image, tf.float32)
    return delf(
        image=float_image,
        score_threshold=tf.constant(100.0),
        image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
        max_feature_num=tf.constant(1000)
    )

def match_images(image1, image2, result1, result2):
    distance_threshold = 0.8

    # Read features
    num_features_1 = result1['locations'].shape[0]
    num_features_2 = result2['locations'].shape[0]

    # Find nearest-neighbor matches using a KD tree
    d1_tree = cKDTree(result1['descriptors'])
    _, indices = d1_tree.query(
        result2['descriptors'],
        distance_upper_bound=distance_threshold
    )

    # Select feature locations for putative matches
    locations_2_to_use = np.array([
        result2['locations'][i,] for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        result1['locations'][indices[i],] for i in range(num_features_2)
        if indices[i] != num_features_1
    ])

    # Perform geometric verification using RANSAC
    _, inliers = ransac(
        (locations_1_to_use, locations_2_to_use),
        AffineTransform,
        min_samples=3,
        residual_threshold=20,
        max_trials=1000
    )

    # Visualize correspondences
    inlier_idxs = np.nonzero(inliers)[0]
    fig, ax = plt.subplots()
    plot_matches(
        ax, image1, image2, locations_1_to_use, locations_2_to_use,
        np.column_stack((inlier_idxs, inlier_idxs)),
        matches_color='b'
    )
    ax.axis('off')
    ax.set_title('DELF correspondences')

    # Convert Matplotlib figure to HTML string
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    correspondences_html = '<img src="data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8') + '" />'
    plt.close()

    return correspondences_html


if __name__ == '__main__':
    app.run(debug=True)
