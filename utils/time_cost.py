import torch
import torch.nn as nn
import os
import tqdm
import torchvision.transforms as transforms
from PIL import Image
from model import Network


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# read_data
def get_image_info(data_path, testsize):
    img_transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.430, 0.387], [0.239, 0.225, 0.230])])
    image = rgb_loader(data_path)
    image = img_transform(image).unsqueeze(0)
    return image

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./pre_train/Net_epoch_best_IIDdata.pth')
    parser.add_argument('--data_path', type=str, default='')
    opt = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    image_names = sorted(os.listdir(opt.data_path))[:100]
    image_tensor = []
    for name in image_names:
        image_path = os.path.join(opt.data_path, name)
        image = get_image_info(image_path, opt.imgsize)
        image_tensor.append(image)
    

    model = Network()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(opt.pth_path))
    model.to(device)
    model.eval()

    random_input = torch.randn(1, 3, 352, 352).to(device)
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(50):
            _, _ = model(random_input)

    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    times = torch.zeros((len(image_tensor), 1))
    with torch.no_grad():
        for iter in tqdm.tqdm(range(len(image_tensor))):
            image = image_tensor[iter]
            starter.record()
            _, _ = model(image)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) 
            times[iter] = curr_time

    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))