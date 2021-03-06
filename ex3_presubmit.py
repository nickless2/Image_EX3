import os, sys, traceback
import numpy as np
from scipy.misc import imread
from skimage import color
import current.sol3 as sol3

def read_image(filename, representation):
    im = imread(filename)
    if representation == 1 and im.ndim == 3 and im.shape[2] == 3:
        im = color.rgb2gray(im)
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.0
    return im

def presubmit():
    print ('ex3 presubmission script')
    disclaimer="""
    Disclaimer
    ----------
    The purpose of this script is to make sure that your code is compliant
    with the exercise API and some of the requirements
    The script does not test the quality of your results.
    Don't assume that passing this script will guarantee that you will get
    a high grade in the exercise
    """
    print (disclaimer)
    
    print('=== Check Submission ===\n')
    if not os.path.exists('current/README'):
        print ('No readme!')
        return False
    with open ('current/README') as f:
        lines = f.readlines()
    print ('login: ', lines[0])
    print ('submitted files:\n' + '\n'.join(map(lambda x: x.strip(), lines[1:])))
    
    print('\n=== Answers to questions ===')
    for q in [1,2,3]:
        if not os.path.exists('current/answer_q%d.txt'%q):
            print ('No answer_q%d.txt!'%q)
            return False
        print ('\nAnswer to Q%d:'%q)
        with open('current/answer_q%d.txt'%q) as f:
            print (f.read())
    
    print ('\n=== Section 3.1 ===\n')
    im_orig = read_image('presubmit_externals/monkey.jpg', 1)
    try:
        print ('Trying to build Gaussian pyramid...')
        gpyr, filter_vec = sol3.build_gaussian_pyramid(im_orig, 3, 3)
        print ('\tPassed!')
        print ('Checking Gaussian pyramid type and structure...')
        if type(gpyr) is not list:
            raise ValueError('Returned pyramid is not a list type. It is %s instead.' % str(type(gpyr)))
        if len(gpyr) != 3:
            raise ValueError('Length of pyramid is wrong. Expecting length 3 list.')
        if filter_vec.shape != (1, 3):
            raise ValueError('Wrong blur filter size. Expecting 1x3')
        if any([l.dtype != np.float32 for l in gpyr]):
            raise ValueError('At least one of the levels in the pyramid is not float32.')
        print ('\tPassed!')
    except:
        print(traceback.format_exc())
        return False

    try:
        print ('Trying to build Laplacian pyramid...')
        lpyr, filter_vec = sol3.build_laplacian_pyramid(im_orig, 3, 3)
        print ('\tPassed!')
        print ('Checking Laplacian pyramid type and structure...')
        if type(lpyr) is not list:
            raise ValueError('Returned pyramid is not a list type. It is %s instead.' % str(type(lpyr)))
        if len(lpyr) != 3:
            raise ValueError('Length of pyramid is wrong. Expecting length 3 list.')
        if filter_vec.shape != (1, 3):
            raise ValueError('Wrong blur filter size. Expecting 1x3')
        if any([l.dtype != np.float32 for l in lpyr]):
            raise ValueError('At least one of the levels in the pyramid is not float32.')
        print ('\tPassed!')
    except:
        print(traceback.format_exc())
        return False

    print ('\n=== Section 3.2 ===\n')
    try:
        print ('Trying to build Laplacian pyramid...')
        lpyr, filter_vec = sol3.build_laplacian_pyramid(im_orig, 3, 3)
        print ('\tPassed!')
        print ('Trying to reconstruct image from pyramid... (we are not checking for quality!)')
        im_r = sol3.laplacian_to_image(lpyr, filter_vec, [1, 1, 1])
        print ('\tPassed!')
        print ('Checking reconstructed image type and structure...')
        if im_r.dtype != np.float32:
            raise ValueError('Reconstructed image is not float32. It is %s instead.' % str(im_r.dtype))
        if im_orig.shape != im_r.shape:
            raise ValueError('Reconstructed image is not the same size as the original image.')
        print ('\tPassed!')
    except:
        print(traceback.format_exc())
        return False

    print ('\n=== Section 3.3 ===\n')
    try:
        print ('Trying to build Gaussian pyramid...')
        gpyr, filter_vec = sol3.build_gaussian_pyramid(im_orig, 3, 3)
        print ('\tPassed!')
        print ('Trying to render pyramid to image...')
        im_pyr = sol3.render_pyramid(gpyr, 2)
        print ('\tPassed!')
        print ('Checking structure of returned image...')
        if im_pyr.shape != (400, 600):
            raise ValueError('Rendered pyramid is not the expected size. Expecting 400x600. Found %s.' % str(im_pyr.shape))
        print ('\tPassed!')
        print ('Trying to display image... (if DISPLAY env var not set, assumes running w/o screen)')
        sol3.display_pyramid(gpyr, 2)
        print ('\tPassed!')
    except:
        print(traceback.format_exc())
        return False

    print ('\n=== Section 4 ===\n')
    try:
        print ('Trying to blend two images... (we are not checking the quality!)')
        im_blend = sol3.pyramid_blending(im_orig, im_orig, np.zeros((400, 400), dtype=np.float32), 3, 3, 5)
        print ('\tPassed!')
        print ('Checking size of blended image...')
        if im_blend.shape != im_orig.shape:
            raise ValueError('Size of blended image is different from the original images and mask used.')
        print ('\tPassed!')
    except:
        print(traceback.format_exc())
        return False
    try:
        print ('Tring to call blending_example1()...')
        im1, im2, mask, im_blend = sol3.blending_example1()
        print ('\tPassed!')
        print ('Checking types of returned results...')
        if im1.dtype != np.float32:
            raise ValueError('im1 is not float32. It is %s instead.' % str(im1.dtype))
        if im2.dtype != np.float32:
            raise ValueError('im2 is not float32. It is %s instead.' % str(im2.dtype))
        if mask.dtype != np.bool:
            raise ValueError('mask is not bool. It is %s instead.' % str(mask.dtype))
        if im_blend.dtype != np.float32:
            raise ValueError('im_blend is not float32. It is %s instead.' % str(im_blend.dtype))
        print ('\tPassed!')
    except:
        print(traceback.format_exc())
        return False

    try:
        print ('Tring to call blending_example2()...')
        im1, im2, mask, im_blend = sol3.blending_example2()
        print ('\tPassed!')
        print ('Checking types of returned results...')
        if im1.dtype != np.float32:
            raise ValueError('im1 is not float32. It is %s instead.' % str(im1.dtype))
        if im2.dtype != np.float32:
            raise ValueError('im2 is not float32. It is %s instead.' % str(im2.dtype))
        if mask.dtype != np.bool:
            raise ValueError('mask is not bool. It is %s instead.' % str(mask.dtype))
        if im_blend.dtype != np.float32:
            raise ValueError('im_blend is not float32. It is %s instead.' % str(im_blend.dtype))
        print ('\tPassed!')
    except:
        print(traceback.format_exc())
        return False

    print ('\n=== All tests have passed ===');
    print ('=== Pre-submission script done ===\n');
    
    print ("""
    Please go over the output and verify that there are no failures/warnings.
    Remember that this script tested only some basic technical aspects of your implementation
    It is your responsibility to make sure your results are actually correct and not only
    technically valid.""")
    return True