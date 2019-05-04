import cv2
import os
import numpy as np

class DataGenerator():
    def __init__(self, image_paths, image_size, batch_size, load_all=False):
        self.image_paths = image_paths
        self.image_size = image_size
        self.batch_size = batch_size
        self.load_all = load_all
        self.generator = self._generator()

    def read_image(self, image_path, resize=False, crop=False):
        image = cv2.imread(image_path)

        if crop:
            image = self.crop_center(image)
            
        if resize:
            image = cv2.resize(image, (self.image_size, self.image_size))

        return image[:, :, ::-1].astype(np.float32) # convert to RGB and float

    def crop_center(self, image):
        h_center, w_center, shift = image.shape[0] // 2, image.shape[1] // 2, self.image_size//2
        
        return image[
            int(h_center-(shift)):int(h_center-(shift)+self.image_size), 
            int(w_center-(shift)):int(w_center-(shift)+self.image_size)
        ]

    def _generator(self):
        if not self.load_all:
            data = np.empty((self.batch_size, self.image_size, self.image_size, 3))
            idxes = np.arange(len(self.image_paths))
            while True:
                np.random.shuffle(idxes)

                i = 0   
                while (i + 1) * self.batch_size <= len(self.image_paths):
                    batch_paths = self.image_paths[i * self.batch_size:(i + 1) * self.batch_size]

                    for j, path in enumerate(batch_paths):
                        img = self.read_image(path, crop=True)
                        data[j] = (img / 127.) - 1
                    i += 1
                    yield data.astype(np.float32)
        else:
            images = np.empty((len(self.image_paths), self.image_size, self.image_size, 3))
            for i, f in enumerate(self.image_paths):
                images[i] = (self.read_image(f) / 127.) - 1
            images = images.astype(np.float32)
            print('loaded')
            
            while True:
                idxes = np.arange(len(self.image_paths))
                np.random.shuffle(idxes)
                images = np.take(images, idxes, axis=0)

                i = 0
                while (i + 1) * self.batch_size <= len(self.image_paths):
                    batch = images[i * self.batch_size:(i + 1) * self.batch_size]
                    if len(batch) < self.batch_size:
                        yield np.concatenate([batch, images[:self.batch_size-len(batch)]], axis=0)
                    else:
                        yield batch

                
if __name__ == '__main__':
    img_size = 128
    dg = DataGenerator([], img_size, 16)
    save_dir = '/home/jovyan/ta-hsi-datacenter/resize_%s_%s/mountain' % (img_size, img_size)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('start download')
    for dirname, dirnames, filenames in os.walk('/home/jovyan/ta-hsi-datacenter/resize_512_512/mountain'):
        for f in filenames:
            image = dg.read_image(os.path.join(dirname, f), resize=True)
            cv2.imwrite(
                os.path.join(save_dir, f),
                (image[:, :, ::-1]).astype(np.uint8)
            )
