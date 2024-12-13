import numpy as np

def get_corners(image):
    height_cut = 40
    width_cut = 40
    image_cut = np.zeros((2*height_cut+40,2*width_cut))
    image_cut[:height_cut,:width_cut] = image[:height_cut,:width_cut]
    image_cut[:height_cut,width_cut:] = image[:height_cut,-width_cut:]
    image_cut[height_cut:2*height_cut,:width_cut] = image[-height_cut:,:width_cut]
    image_cut[height_cut:2*height_cut,width_cut:] = image[-height_cut:,-width_cut:]
    image_cut[2*height_cut:2*height_cut+20,:] = image[:20,1024-width_cut:1024+width_cut]
    image_cut[2*height_cut+20:,:] = image[-20:,1024-width_cut:1024+width_cut]

    return image_cut

def get_lateral_region(image):
    y_center = np.shape(image)[0]//2
    y_min = y_center - 50
    y_max = y_center + 50
    x_min = 100
    x_max = 300
    return image[y_min:y_max,x_min:x_max]