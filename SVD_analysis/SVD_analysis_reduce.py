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

def get_corners_custom(image, vert_size=20, hor_size=100):
    """
    Extract the four corners of the image with configurable size.
    
    Parameters:
    -----------
    image : array
        Input image
    vert_size : int
        Vertical size of corner regions in pixels
    hor_size : int
        Horizontal size of corner regions in pixels
    
    Returns:
    --------
    image_cut : array
        Concatenated corner regions (shape: 2*vert_size, 2*hor_size)
    """
    image_cut = np.zeros((2*vert_size, 2*hor_size))
    # Top-left corner
    image_cut[:vert_size, :hor_size] = image[:vert_size, :hor_size]
    # Top-right corner
    image_cut[:vert_size, hor_size:] = image[:vert_size, -hor_size:]
    # Bottom-left corner
    image_cut[vert_size:, :hor_size] = image[-vert_size:, :hor_size]
    # Bottom-right corner
    image_cut[vert_size:, hor_size:] = image[-vert_size:, -hor_size:]
    
    return image_cut

def get_lateral_region(image):
    y_center = np.shape(image)[0]//2
    y_min = y_center - 50
    y_max = y_center + 50
    x_min = 100
    x_max = 300
    return image[y_min:y_max,x_min:x_max]

def get_two_lateral_regions(image):
    y_center = np.shape(image)[0]//2
    y_min = y_center - 50
    y_max = y_center + 50
    x1_min = 100
    x1_max = 300
    x2_min = np.shape(image)[1] - 300
    x2_max = np.shape(image)[1] - 100
    image_cut = np.zeros((2*50,400))
    image_cut[:,:200] = image[y_min:y_max,x1_min:x1_max]
    image_cut[:,200:] = image[y_min:y_max,x2_min:x2_max]
    return image_cut