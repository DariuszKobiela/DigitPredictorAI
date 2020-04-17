import warnings

from flask import Flask, render_template

from models import load_images
from utils import model_predict, render_image, get_image_from_file

warnings.filterwarnings("ignore")

x, y = load_images()
app = Flask(__name__)
HTML_FILE = 'index.html'


@app.route('/')
def home():
    image_name = render_image(x)
    return render_template(HTML_FILE, image_name=image_name, prediction_value='?')


@app.route('/render', methods=['POST'])
def render():
    image_name = render_image(x)
    return render_template(HTML_FILE, image_name=image_name, prediction_value='?')


@app.route('/predict', methods=['POST'])
def predict():
    image, image_name = get_image_from_file()
    prediction = model_predict(image)
    return render_template(HTML_FILE, image_name=image_name, prediction_value=prediction)


if __name__ == "__main__":
    app.run(debug=False)
