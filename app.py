from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import io
import os
from ultralytics import YOLO
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Global variables
current_video_path = None
camera = None
model = YOLO("best.pt")  # Load model once

@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    global current_video_path, camera
    
    # Close existing camera if open
    if camera is not None:
        camera.release()
    
    # Save the uploaded file
    file_path = f"uploads/{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Update current video path
        current_video_path = file_path
        
        # Test if video can be opened
        test_camera = cv2.VideoCapture(file_path)
        if not test_camera.isOpened():
            raise HTTPException(status_code=400, detail="Invalid video file")
        test_camera.release()
        
        return {"filename": file.filename, "status": "success"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing video: {str(e)}"}
        )

async def generate_frames():
    global current_video_path, camera
    
    if current_video_path is None:
        raise HTTPException(status_code=400, detail="No video uploaded")
    
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(current_video_path)
    
    while True:
        success, frame = camera.read()
        
        if not success:
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            continue
            
        try:
            # Run YOLO detection
            results = model.predict(frame, device='cpu')
            frame = results[0].plot()  # Plot the detection results
            
            # Add any additional visualization
            cv2.rectangle(frame, (10, 5), (40, 300), (255, 0, 0), 2)
            
            # Convert to bytes
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
