from flask import Flask
from info_process import app_data
import os

main_app = Flask(__name__)
main_app.register_blueprint(app_data)

main_app.secret_key = os.urandom(24)
main_app.run(debug=True, port=8000)