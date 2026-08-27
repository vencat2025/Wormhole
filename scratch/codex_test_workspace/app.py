from flask import Flask, render_template_string
import os

app = Flask(__name__)

@app.route('/')
def display_image():
    try:
        # Ensure the image exists
        if not os.path.exists('image.jpg'):
            raise FileNotFoundError("Image file not found")

        # Read the image data and encode it in base64
        with open('image.jpg', 'rb') as image_file:
            encoded_string = image_file.read().encode('base64').decode('utf-8')

        # HTML to display the image
        html_content = f'<img src="data:image/jpeg;base64,{encoded_string}" alt="Image"/>'

        return render_template_string(html_content)
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
