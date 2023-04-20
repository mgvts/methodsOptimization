import os

import imageio as imageio

path = 'images/temp'
frames = []
print(sorted(os.listdir(path), key=lambda i: int(i.split('_')[1].replace('.png', ''))))
for i in (sorted(os.listdir(path), key=lambda i: int(i.split('_')[1].replace('.png', '')))):
    image = imageio.v2.imread(f'{path}/{i}')
    frames.append(image)


imageio.mimsave('./example.gif',
                frames,
                fps = 5,
                loop = 1
                )
