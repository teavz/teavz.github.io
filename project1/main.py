# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
import skimage.transform as sktr

def ncc(im1, im2):
    im1 = im1 - np.mean(im1)
    im2 = im2 - np.mean(im2)
    
    norm1 = np.linalg.norm(im1)
    norm2 = np.linalg.norm(im2)
    
    if norm1 == 0 or norm2 == 0:
        return -np.inf
    
    return np.sum(im1 * im2) / (norm1 * norm2)

def single_align(im, ref, shift=15):
    best_score = -np.inf
    best_shift = (0, 0)
    h, w = im.shape
    
    #ignore edges
    crop_x = int(0.1 * w)
    crop_y = int(0.1 * h)
    
    for dx in range(-shift, shift + 1):
        for dy in range(-shift, shift+1):
            shifted_img = np.roll(im, shift = (dx, dy), axis=(0, 1))
            
            ref_center = ref[crop_y:-crop_y, crop_x:-crop_x]
            shifted_center = shifted_img[crop_y:-crop_y, crop_x:-crop_x]
            
            score = ncc(ref_center, shifted_center)
            if score > best_score:
                best_score = score
                best_shift = (dx, dy)
    
    return np.roll(im, shift=best_shift, axis=(0,1)), best_shift
def multi_align(im, ref, shift=15):

    h, w = im.shape
    total_shift = (0, 0)
    if min(h, w) >= 500:
        im_small = sktr.rescale(im, 0.8, anti_aliasing = True)
        ref_small = sktr.rescale(ref, 0.8, anti_aliasing = True)
        
        _, coarse_shift = multi_align(im_small, ref_small, shift)
        
        coarse_shift = (round(coarse_shift[0] / 0.8), round(coarse_shift[1] / 0.8))
        im = np.roll(im, shift=coarse_shift, axis=(0, 1))
        total_shift = coarse_shift
        shift = 15
        
    best_score = -np.inf
    best_shift = (0, 0)
    h, w = im.shape
    
    #ignore edges
    crop_x = int(0.1 * w)
    crop_y = int(0.1 * h)
    
    for dx in range(-shift, shift + 1):
        for dy in range(-shift, shift+1):
            shifted_img = np.roll(im, shift = (dx, dy), axis=(0, 1))
            
            ref_center = ref[crop_y:-crop_y, crop_x:-crop_x]
            shifted_center = shifted_img[crop_y:-crop_y, crop_x:-crop_x]
            
            score = ncc(ref_center, shifted_center)
            if score > best_score:
                best_score = score
                best_shift = (dx, dy)
    
    total_shift = (total_shift[0] + best_shift[0], total_shift[1] + best_shift[1])
    return np.roll(im, shift=best_shift, axis=(0,1)), total_shift

def main():
    # name of the input file
    imname = 'CS180_fa2026_proj1_data/three_generations.tif'
    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(np.int_)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # align the images
    ag, best_green= multi_align(g, b)
    ar, best_red= multi_align(r, b)

    print(imname)
    print("best green: ", best_green)
    print("best red: ", best_red)
    # create a color image
    im_out = np.dstack([ar, ag, b])

    # save the image
    im_out_uint8 = sk.img_as_ubyte(im_out)
    fname = 'output/multi_three_generations.jpg'
    skio.imsave(fname, im_out_uint8)

    # display the image
    plt.imshow(im_out)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()