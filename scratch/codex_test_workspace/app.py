from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route('/image')
def show_image():
    return send_from_directory('.', 'example.jpg', mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
