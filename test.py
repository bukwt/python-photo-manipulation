from image import *

#Test functions for all functionalities


def test_brighten():

    img = Image(filename="data/ist.png")
    img.change_brightness(50)
    img.save_image("data/brighter.png")


def test_darker():
    img = Image(filename="data/ist.png")
    img.change_brightness(-50)
    img.save_image("data/darker.png")


def test_contrast_lower():

    img = Image(filename="data/ist.png")
    img.change_contrast(0.7)
    img.save_image("data/lower-contrast.png")


def test_contrast_higher():

    img = Image(filename="data/ist.png")
    img.change_contrast(1.5)
    img.save_image("data/higher-contrast.png")


def test_blending():
    img = Image(filename="data/ist.png")
    img.change_contrast(1.5)

    img_2 = Image(filename="data/ist.png")
    img_2.change_contrast(0.7)

    #higher contrast img blended with lower contrast img
    img.blend(img_2,0.6)
    img.save_image("data/blended.png")

def test_black_and_white():
    
    #rgb test img for black and white
    img = Image(filename="data/ist.png")
    img.black_and_white(110)
    img.save_image("data/black_and_white.png")

def test_blur():
    #blur image
    blur_img = Image(filename="data/ist.png")
    blur_img.blur(5)
    blur_img.save_image("data/blur.png")

def test_gaussian_blur():

    #gaussian blur image
    gauss_img= Image(filename="data/ist.png")
    gauss_img.gaussian_blur(5,2)
    gauss_img.save_image("data/gauss-blur.png")


def test_edge_detection():
    edge_img = Image(filename="data/ist.png")
    edge_img.find_edges()
    edge_img.save_image("data/derivative.png")


if __name__=="__main__":
    #Calls for the test functions
    
    test_brighten()
    test_darker()
    test_black_and_white()
    test_contrast_higher()
    test_contrast_lower()
    test_blending()
    test_blur()
    test_gaussian_blur()
    test_edge_detection()

   
   

   

  