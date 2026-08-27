from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Image Display</title>
        </head>
        <body>
            <img src="https://via.placeholder.com/150" alt="Placeholder Image">
        </body>
        </html>
    ''')

if __name__ == '__main__':
    app.run(debug=True)
