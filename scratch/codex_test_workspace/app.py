# app.py
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    # Path to your image file relative to the static folder
    image_url = '/static/your_image.jpg'
    return render_template('index.html', image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
