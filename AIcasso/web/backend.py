from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Form
from fastapi.responses import JSONResponse
import io
from PIL import Image
import sys
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set the working directory to HairFastGAN to ensure relative paths are correct
os.chdir('/home/user/AIcasso/AIcasso/HairFastGAN')

# Print current working directory
print("Current working directory:", os.getcwd())

# Add the HairFastGAN directory to sys.path to allow imports
sys.path.append(os.getcwd())

print("sys.path:", sys.path)

# Use absolute imports
from main import process_image
from hair_swap import get_parser

app = FastAPI()

# Use relative paths within the HairFastGAN directory
USER_IMAGE_PATH = "input/user_image.png"
RESULT_IMAGE_PATH = "output/user_result.png"
MODEL_PATH = "."

# Initialize the model arguments once
model_parser = get_parser()
model_args, _ = model_parser.parse_known_args()

@app.post("/upload/")
async def create_upload_file(
    file: UploadFile = File(...),
    color_image_name: str = Form(None),
    shape_image_name: str = Form(None),
    mode: str = Form("both")
):
    try:
        print(f"Received file: {file.filename}")
        print(f"Color image name: {color_image_name}")
        print(f"Shape image name: {shape_image_name}")
        print(f"Mode: {mode}")

        # Load the user image from the uploaded file
        image = Image.open(io.BytesIO(await file.read()))

        # Save the user image as 'user_image.png' in the specified directory
        image.save(USER_IMAGE_PATH, format="PNG")

        # Determine paths for the shape and color images
        shape_path = Path(MODEL_PATH) / "input" / shape_image_name if shape_image_name else None
        color_path = Path(MODEL_PATH) / "input" / color_image_name if color_image_name else None

        # Ensure previous result is deleted if it exists
        if os.path.exists(RESULT_IMAGE_PATH):
            os.remove(RESULT_IMAGE_PATH)

        # Call the process_image function
        process_image(
            face_path=Path(USER_IMAGE_PATH),
            shape_path=shape_path,
            color_path=color_path,
            output_path=Path(RESULT_IMAGE_PATH),
            model_args=model_args,
            benchmark=False
        )

        # Check if the result image exists and return it
        if os.path.exists(RESULT_IMAGE_PATH):
            processed_image = Image.open(RESULT_IMAGE_PATH)
            buf = io.BytesIO()
            processed_image.save(buf, format="JPEG")
            buf.seek(0)
            return Response(content=buf.getvalue(), media_type="image/jpeg")
        else:
            raise FileNotFoundError(f"Result image not found at {RESULT_IMAGE_PATH}")

    except Exception as e:
        return JSONResponse(status_code=422, content={"message": f"Error processing image: {e}"})
