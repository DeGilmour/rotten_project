from flask import Flask
from dotenv import load_dotenv
from rotten_project.controllers import controller


load_dotenv()
app = Flask(__name__)
app.register_blueprint(controller.alfredo)

if __name__ == "__main__":
    app.run(debug=True)