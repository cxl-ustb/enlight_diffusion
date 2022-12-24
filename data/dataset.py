from torch.utils.data import Dataset
import os
import torch,torchvision
from PIL import Image

totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
def transform_augment(img_list, split='val', min_max=(0, 1)):    
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)

class HL_light_Dataset(Dataset):
    def __init__(self,dataroot):
        self.split='train'
        self.dataroot=dataroot
        self.low_light_path=get_paths_from_images(
            os.path.join(self.dataroot,'low')
        )
        self.high_light_path=get_paths_from_images(
            os.path.join(self.dataroot,'high')
        )

        assert len(self.low_light_path)==len(self.high_light_path)
        self.data_len=len(self.low_light_path)
        
    def __len__(self,):
        return self.data_len
    
    def __getitem__(self, index):
        img_l_light=Image.open(self.low_light_path[index]).convert('RGB')
        img_h_light=Image.open(self.high_light_path[index]).convert('RGB')
        [img_l_light, img_h_light]=transform_augment(
                [img_l_light, img_h_light], split=self.split, min_max=(-1, 1))
        return {'low_light':img_l_light,'high_light':img_h_light,'index':index}
        
        

if __name__ == '__main__':
    data=HL_light_Dataset('/data/dataset/lol/our485_256_256')
    
    print(data[0]['low_light'].shape)