from flask import Flask, send_file

app = Flask(__name__)

@app.route('/')
def show_image():
    return send_file('path/to/image.jpg', mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
