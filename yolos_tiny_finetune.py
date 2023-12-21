from transformers import YolosImageProcessor, YolosForObjectDetection, get_scheduler
import yolos_tiny_dataloader as yd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch import stack, save


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = stack(images, dim=0)
    return images, targets


model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
dataloader = DataLoader(yd.YOLODataset(directory="./headcrab_dataset"), batch_size=4, shuffle=True, collate_fn=collate_fn)
print(dataloader.dataset[10])

# print(list(model.parameters()))
# yolo_data = yd.YOLODataset(directory="./headcrab_dataset")
# print(len(yolo_data))
# print(yolo_data[0])
# exit()
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs_num = 50
training_steps = epochs_num * len(dataloader)
print(training_steps)
lr_scheduler = get_scheduler(name='linear',
                             optimizer=optimizer,
                             num_warmup_steps=10,
                             num_training_steps=training_steps)

progress_bar = tqdm(range(training_steps))
model.train()
for epoch in range(epochs_num):
    for batch in dataloader:
    #    print(batch)
       images, targets = batch
       images = images.to('cpu')
    #    batch = {k: v.to("cpu") for k, v in batch.items()}
       targets = [{k: v.to("cpu") for k, v in t.items()} for t in targets]
    #    outputs = model(**batch)
       outputs = model(images, targets)
       loss = outputs.loss
       print(loss.item())
       loss.backward()
       optimizer.step()
       lr_scheduler.step()
       optimizer.zero_grad()
       progress_bar.update(1) 

save(model.state_dict(), "yolo-headcrabs.pth")