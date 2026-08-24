# app.py
from flask import Flask, render_template, url_for

app = Flask(__name__)

@app.route('/')
def index():
    # Assuming the image is placed in the static folder as 'images/example.jpg'
    image_url = url_for('static', filename='images/example.jpg')
    return f'''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Image Display</title>
      </head>
      <body>
        <h1>Here is your image</h1>
        <img src="{image_url}" alt="Example Image">
      </body>
    </html>
    '''

if __name__ == "__main__":
    app.run(debug=True)
