from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch
import pprint as pp
# import requests

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
file = "939fcea1-20231218183427_1.jpg"
image = Image.open(file)

model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
model.load_state_dict(torch.load('yolo-headcrabs.pth'))
# pp.pprint(model.config.id2label)
model.config.id2label.update({0: 'classic headcrab', 2: "poison headcrab"})

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# model predicts bounding boxes and corresponding COCO classes
logits = outputs.logits
bboxes = outputs.pred_boxes


# print results
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(label)
    found_object = model.config.id2label[label.item()]
    conf = round(score.item(), 3)
    print(
        f"Detected {found_object} with confidence "
        f"{conf} at location {box}"
    )
    print(type(box))
    draw = ImageDraw.Draw(image)
    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline='cyan')
    draw.text((box[0] + 5, box[1] + 5), f"{found_object}: {conf}")
    image.save(f"{file.split('.')[0]}_predict.jpg", 'jpeg')