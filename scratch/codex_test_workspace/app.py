from flask import Flask, render_template_string
app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string('<img src="static/image.jpg" alt="Image">')

if __name__ == '__main__':
    app.run(debug=True)
