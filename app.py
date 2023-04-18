from fastapi import FastAPI, Request
from fastapi import UploadFile, File
from backend.prediction_engine import get_yolov5, get_image_from_bytes, pipe, read_input
from starlette.responses import Response
from PIL import Image
import json
import io
import numpy as np
import cv2
from torchvision import transforms
from pydantic import BaseModel
from werkzeug.utils import secure_filename
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

model = get_yolov5()
app = FastAPI(title="Car Dekhliya")

templates = Jinja2Templates(directory="backend/static")

@app.get("/")
def write_home(request: Request):
    return templates.TemplateResponse("File-Upload.html", {"request":request})

@app.post("/object-to-json")
async def detect_food_return_json_result(file: UploadFile = File(...)):
    input_image = await file.read()
    results = model(get_image_from_bytes(input_image))
    detect_res = results.pandas().xyxy[0].to_json(orient="records")
    detect_res = json.loads(detect_res)
    return {"result": detect_res}



@app.post("/object-to-img")
async def detect_food_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    for img in results.ims:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(),
media_type="image/jpeg")

@app.post('/severity')
async def predict_severity(file: UploadFile = File(...)):
    image = await file.read()
    filename = secure_filename(file.filename)
    model_results = pipe(filename)
    return JSONResponse(content={"result": model_results}, status_code=200)