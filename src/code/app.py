from flask import Flask
from flask import request
from flask import make_response
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from werkzeug.utils import secure_filename
import os
import glob
from flask import render_template

app = Flask(__name__, 
            static_url_path='', 
            static_folder='public',)
img_cartoon = pipeline(Tasks.image_portrait_stylization, 
                       model='damo/cv_unet_person-image-cartoon-3d_compound-models')
# model = Model.from_pretrained("./models/cv_unet_person-image-cartoon-3d_compound-models")
# img_cartoon = pipeline(Tasks.image_portrait_stylization, 
#                        model=model)

CACHE_DIR = "/etc/image-stylization/img_cache"

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/api/image-stylization", methods=["POST"])
def image_stylization():
    file = request.files['file']
    filename = secure_filename(file.filename)
    files_to_remove = glob.glob(os.path.join(CACHE_DIR, "*"))
    for f in files_to_remove:
        os.remove(f)
    file.save(os.path.join(CACHE_DIR, filename))
    result = img_cartoon(os.path.join(CACHE_DIR, filename))
    retval, buffer = cv2.imencode('.png', result[OutputKeys.OUTPUT_IMG])
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/png'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response