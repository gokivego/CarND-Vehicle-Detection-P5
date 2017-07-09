import os
import glob

def loadImages(directory, fileName):

    folders = os.listdir(directory)
    images = []
    for folder in folders:
        images.extend(glob.glob(directory + folder +'/*'))
    print ('Number of images in the {} directory = {}'.format(directory ,len(images)))
    with open(fileName + "txt", 'w') as f:
        for fn in images:
            f.write(fn + '\n')
    f.close()

    return images