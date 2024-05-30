import PIL.Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
from pathlib import Path
import cv2

video_file = Path("../data/input/original_video.mp4")

face_draw_dir = Path("../data/face/")
if not face_draw_dir.exists():
    face_draw_dir.mkdir()

split_face_dir = Path("../data//split_face/")
if not split_face_dir.exists():
    split_face_dir.mkdir()

# %%
mtcnn = MTCNN()


def get_boxes(img: PIL.Image.Image):
    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        return None
    return [list(box) for box in boxes]


# %%

cap = cv2.VideoCapture(video_file.as_posix())

FRAME_INTERVAL = 120

face_idx = 0
i = 0
while True:
    i += 1
    # ret is a boolean variable that returns true if the frame is available.
    ret, frame = cap.read()

    if not ret:
        break

    if i % FRAME_INTERVAL != 0:
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    boxes = get_boxes(img)
    if boxes is None:
        continue
    else:
        for box in boxes:
            face = img.crop(box)
            face.save(f"../data/split_face/{face_idx}.jpg")
            face_idx += 1
            draw.rectangle(box, outline=(255, 0, 0), width=2)
        img.save(f"../data/face/{i}.jpg")

    # break
