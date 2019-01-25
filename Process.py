from core.backend import ModelWrapper
import cv2

model_wrapper = ModelWrapper()

img = cv2.imread('input.jpg')
image_data = cv2.imencode('.jpg', img)[1].tostring()

# with open('input.jpg', 'rb') as f:
#     image_data = f.read()

preds = model_wrapper.predict(image_data)

label_preds = [{'index': p[0], 'caption': p[1], 'probability': p[2]} for p in preds]

print(label_preds)

