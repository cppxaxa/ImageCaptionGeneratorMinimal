from core.backend import ModelWrapper

model_wrapper = ModelWrapper()

image_data = None
with open('input.jpg', 'rb') as f:
    image_data = f.read()
    
preds = model_wrapper.predict(image_data)

label_preds = [{'index': p[0], 'caption': p[1], 'probability': p[2]} for p in preds]

print(label_preds)

