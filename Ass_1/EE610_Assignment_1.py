# Import required modules
# Tkinter for GUI
from tkinter import *
# filedialog for handling file operations: Loading and saving the image
from tkinter import filedialog
from tkinter.filedialog import askopenfilename, asksaveasfilename
# PIL for Image operations
from PIL import Image, ImageTk
# Numpy for array operations
import numpy as np      # using np as whole numpy
# math for math operations: such as exponentials, logs
import math

# Tkinter Object:
root = Tk()     # root: Tkinter Object
root.title("Basic Image Editor")        #Setting the tittle of GUI as Basic Image Editor
# Changing Background colour
root.configure(bg='#8676AD')        # #86876AD is colour code for Purple
# Setting Geometry of GUI frame:
root.geometry("1500x840")       # 1500 x 840 will be the default size of window
label_title = Label(root, text="Image Processing (EE610): Basic Image Editor", font=('ariel', 20), bg='#8676AD')        #label_title is a variable to place title on the top of program
label_title.place(x=585, y=2)       # Placing the label_title at middle of the GUI

# Creating Global Variables:
global imag, im_type, stack, pathh      # stack for undo operation, im_type: whether color or grescale, imag: global variable for original image, pathh: global variable for path of original image
# Using Stacks for undo and reset operation
stack = []


# Functions for General File Operation:
# 1. Loading the Image:
def selects():
    """
    Opens a file dialog box and loads the image in the GUI
    """

    global imag, im_type, stack, pathh      # Using global command so that the global variables could be used inside the function
    # Open Image using File selector
    pathh = filedialog.askopenfilename()        # taking path after asking the user from file selector
    imag = Image.open(pathh)        # Open the image from path and store into global variable imag
    # Resizing Image so that it won't cross our GUI frame
    imag.thumbnail((1230, 530))
    # Conversion from RGB to HSV and storing in an numpy array with updating the image type
    if (imag.mode == "RGB" or imag.mode == "RGBA" or imag.mode == "CMYK" or imag.mode == "LAB" or imag.mode == "HSV"):
        im_type = "color"       # Initializing im_type variable as colour
        pix = np.array(imag.convert('HSV'))         #Converting color RGB image to HSV and creating np.array
    else:
        im_type = "grey"        # Initializing im_type variable as greyscale image 'grey'
        pix = np.array(imag.convert('L'))       # Making the np.array of image by converting all greyscale images ito one specific format

    # displaying the image in the small canvas
    imag1 = Image.open(pathh)       # Using Different variable, imag1 to show the image into small canvas
    imag1.thumbnail((228, 128))     # Resizing the image, imag1 according to the small canval
    imag2 = ImageTk.PhotoImage(imag1)       # creating imag2 as ImageTk object
    canva2.create_image(120, 70, image=imag2)       # Creating image at small canvas
    canva2.image = imag2        # Showing the Image at small canvas
    # Clear the stack everytime a new image is loaded
    stack.clear()
    # Put the new Image array into the stack
    stack.append(pix)
    # Display the Image into the main canvas
    Display(pix)


# 2. Function to display the image:
def Display(arr):
    '''
    Takes a numpy array and displays the Image into main Canvas
    '''

    if (im_type == "color"):
        # Read Image from numpy array as coloured and HSV as that is how it is stored
        imag1 = Image.fromarray(arr, 'HSV')
    else:
        # Read Image from numpy array as greyscale image
        imag1 = Image.fromarray(arr)

    # Resize image for displaying on the canvas
    imag1.thumbnail((1230, 530))
    imag2 = ImageTk.PhotoImage(imag1)       # Creating imag2 as ImakeTk object
    # Putting the image into Canvas
    canva.create_image(580, 325, image=imag2)       # Creating image at main canvas
    canva.image = imag2     # Displaying image at main canvas


# 3. Function for Save the modified Image:
def Savee():
    '''
    Saves the last modified Image at a user defined location
    '''

    global stack, im_type, pathh, imag      # global variables so that we can use them in this function
    # Location where to save:
    # Default Location: location where oroginal Image Exists
    deff = pathh.split(".")[-1]
    # Asking user for location
    where_to = asksaveasfilename(defaultextension=f".{deff}", filetypes=[("All Files", '*,*'),
                                                                         ("PNG file", "*.png"), ("jpg file", "*.jpg")])
    if (im_type == "color"):
        # For Coloured Image: read from array convert to RGB and save
        imag1 = Image.fromarray(stack[-1], 'HSV')       # imag1 is image object created from the numpy array
        imag2 = imag1.convert('RGB')        # imag1 is now converted into RGB and stored in imag2
    else:
        # For greyscale Image: read from array and save
        imag2 = Image.fromarray(stack[-1])      # Using the same name for image object of grescale after converting it from numpy array as imag2
    imag2.save(where_to)        # Saving the imag2 at the location specified by user

# 4. Function for Reset to original Image:
def Resett():
    '''
    Resetts the modigied image into original Image
    '''
    global stack        # global variables so that we can use them in this function
    # Copy the first element from stack that is the original image
    pix = np.copy(stack[0])
    # Clear the stack and Display original Image in Canvas
    stack.clear()
    stack.append(pix)       # Put the original image into stack after clearing
    Display(pix)        # Display the original image into main canvas


# 5. Function for Undo Operation:
def Undoo():
    '''
    Carries out undo operation
    '''

    global stack        # global variables so that we can use them in this function
    # if length of stack > 1 then only delete the last element else not
    if (len(stack) > 1):
        stack.pop()     # Pop: deletes the last inserted element from stack
        Display(stack[-1])      # Display the image after pop operation
    else:
        Display(stack[-1])      # if image is equal to original image then diplay original image


# Functions for modifying Image:
# 6. Function for Histogram Equalization:
def Histogramm():
    '''
    Applies Histogram Eqalization to the last modified Image
    '''

    global stack, im_type       # global variables so that we can use them in this function
    # Copy the last modified image
    pix = np.copy(stack[-1])        # pix is used to copy the last numpy array from stack
    # Take the Value component for Coloured Image and whole image for Greyscale Image
    if (im_type == "color"):
        V = np.rollaxis(pix, -1)[2]     # Creating numpy variable for V component
    else:
        V = pix     # Using the same name (V) for greyscale images also for computational advantages
    # Using a list hist[] for storing the number of pixels at every intensity level (0-255)
    hist = []
    # Using a dictionary for storing the transformation
    s = {}      # s: a dictionary to store the transformation
    # count number of pixels for each intensity level
    for i in range(0, 256):     # Running a loop over all pixel intensities assuming there are 256 pixes for all image
        hist.append(np.count_nonzero(V == i) / (V.shape[0] * V.shape[1]))       # updating the list with numbers of pixels in each intensity level
        if (i > 0):     # for updating the non first element of dictionary
            s[i] = (s[i - 1] + hist[i])     # non first elements are updating as this equation
        else:
            s[i] = (hist[i])        # for updating the first element of dictionary
    # Updating all the pixels with intensity r to intensity s
    for i in range(0, 256):
        # Round normalised s_i after multiplying with 255
        s[i] = round(s[i] * 255)
        # Replace all the intesity of i with s[i]
        V = np.where(V == i, s[i], V)
        V=np.round(V)       #Rounding off to nearest integer for storing as array so that conversion to image is easy
    # Update V component in image or greyscale image and store in stack
    if (im_type == "color"):
        pix[:, :, 2] = V        # Putting the V component back in pix
    else:
        pix = V     # changing the pix array to modified array
    stack.append(pix)       # updating the stack with last modified image array
    # Display Updated Image in canvas
    Display(stack[-1])      # Displaying the last modified image into the main cancas


# 7. Function for Gamma Transform:
def Gammaa():
    '''
    Applies Gamma Transformation to the last modified Image
    '''

    global stack, im_type, gammm        # global variables so that we can use them in this function
    # Creating Label and Entry for Taking the value of Gamma from User
    label1 = Label(root, text='Please Enter Value of gamma:', bg='#E0E0E0')     # label1 to ask user for gamma value
    label1.place(x=30, y=475)       # placing the label1 at appropriate location in the GUI
    e = Entry(root, width=3, bg='#E0E0E0')      # e: Entry object for taking the input from user
    e.place(x=194, y=476)       # placing e at appropriate location in the GUI

    # 7.1. Sub function for Gamma Transform- For Button OK:
    def Okk():
        # Defining Global Gamma so that it could be used outside of function
        global gammm        # gammm is used for the value of Gamma
        # Getting the value from Entry module after user has put the value
        gammm = float(e.get())      # updating the gammm after the user had entered the value
        # Repeating the steps from Histogramm function to get the V component of last modified Image
        pix = np.copy(stack[-1])
        if (im_type == "color"):
            V = np.rollaxis(pix, -1)[2]
        else:
            V = pix

        # Value of Normalization constant as c
        c = 255 / math.pow(255, gammm)
        # Applying the Gamma transform
        V = c * (np.power(V, gammm))
        V = np.array(np.round(V))       # Rounding off to nearest integer for storing as array so that conversion to image is easy

        # Repeating the steps from Histogram function to store and display the image
        if (im_type == "color"):
            pix[:, :, 2] = V
        else:
            pix = V
        stack.append(pix)
        Display(pix)

        # Destroying thelabels and buttons so that other functions can use that area in GUI
        label1.after(100, label1.destroy)       # destroy the label1 after use
        e.after(100, e.destroy)     # destroy the entry object after use
        okk.after(100, okk.destroy)     # destroy the okk button after use

    # okk button that allows the program to know that user has entered the value of Gamma
    okk = Button(root, text='OK', font=('ariel', 7), bg='#E0E0E0', command=Okk)     # okk button for user to press after entering the gamma value
    okk.place(x=220, y=475)     # placing the okk button at appropriate location


# 8. Function for Log Transform:
def Llog():
    '''
    Applies Log Transformation to the last modified Image
    '''

    # Repeating the steps from Histogramm function to get the V component of last modified Image
    global stack, im_type
    pix = np.copy(stack[-1])
    if (im_type == "color"):
        V = np.rollaxis(pix, -1)[2]
    else:
        V = pix
    # Normalization Constant c
    c = 255 / np.log(1 + 255)
    # Applying the Log Transform
    V = c * (np.log(V + 1))     # adding 1 because log(0) is not defined
    V = np.array(np.round(V))       # Rounding off to nearest integer

    # Repeating the steps from Histogram function to store and display the image
    if (im_type == "color"):
        pix[:, :, 2] = V
    else:
        pix = V
    stack.append(pix)
    Display(pix)


# 9. Function for Vectorised Convolution:
def convo(img, fil):
    '''
    Takes two numpy arrays as input: img and fil and returns the 
    result after convolution
    '''
    # Filter Hight and Width are same and equal to the size of filter
    fil_hw = fil.shape[0]
    # Calculating and storing the shape of Convolution Result
    out_shape = np.array(img.shape) - np.array(fil.shape) + 1
    # Creating the i and j values to broadcast the original 
    #image so that we can apply vectorised 
    #Implementation of Convolution
    i0 = np.repeat(np.arange(fil_hw), fil_hw)                            # This operation is explained in detail in the report
    j0 = np.tile(np.arange(fil_hw), fil_hw)                              # This operation is explained in detail in the report
    i1 = np.repeat(np.arange(out_shape[0]), out_shape[1])                # This operation is explained in detail in the report
    j1 = np.tile(np.arange(out_shape[1]), out_shape[0])                  # This operation is explained in detail in the report
    # Broadcasting i0, i1, j0, j1 to get i and j
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)                            # This operation is explained in detail in the report
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)                            # This operation is explained in detail in the report

    # Applying i and j to get broadcasted image
    image_rec = img[i, j]
    # Flatten the Filter
    fil_flat = fil.reshape(fil_hw * fil_hw, -1)
    # Apply dot product to get convoluted image
    convol_res = np.dot(fil_flat.T, image_rec)
    # Reshape the Output
    convol_res = convol_res.reshape(out_shape[0], out_shape[1])
    return convol_res


# 12. Blurring Operation:

# 12.1. Function to create gaussian filter of size mxm
def gaussian_filter(win_size, sigma):
    '''
    Takes window size and sigma as input and returns Gaussian filter of that size and sigma
    '''
    # Initialize filter with numpy zeros
    fil = np.zeros((win_size, win_size))
    # Defining Origin at the center of filter
    origin = win_size // 2
    # Running Loop for all pixel
    for i in range(fil.shape[0]):
        for j in range(fil.shape[1]):
            # Distance from the origin or the power of exponential
            dist = ((i - origin) ** 2 + (j - origin) ** 2) / (2 * sigma ** 2)
            # Calculating values at each i and j
            fil[i][j] = math.exp(-(dist)) / (math.sqrt(2 * 3.14) * sigma)

    # Returning the Normalized values
    return fil / fil.sum()


# 12.2. Function for Box Filter:
def box_filter(win_size):
    '''
    takes window size as parameters and returns normalised box filter
    '''
    # return normalised np.ones of size win_size x win_size
    return (np.ones((win_size, win_size), int) / (win_size * win_size))


# 12.3. Choosing the type of filter:
def choose_filter():
    '''
    Makes Buttons for choosing the filter and calls create_button_blur function
    with parameter filter type
    '''

    # Making buttons and labels global variables so that they can be destroyed later
    global butt_box, butt_gauss, label_filter       # butt_box for creating button for box filter, butt_gauss: for creating gaussian filter, label_filter: to let the user know for choosing a filter
    # Creating Label to ask user for filter type
    label_filter = Label(root, text='Please Choose the type of filter: ', font=('ariel', 10), bg='#E0E0E0')     # label_filter to ask the user for filter type
    label_filter.place(x=30, y=455)     # Placing the label_filter at appropriate location

    # Defining functions for Buttons
    # Function for box filter
    def but_box():
        create_button_blur('box')       # Calling the function with box filter as parameter

    # Function for Gaussian filter
    def but_gauss():
        create_button_blur('gauss')     # calling the functions as gaussian filter as parameter

    # Creating Button for Box Filter
    butt_box = Button(root, text='Box Filter', font=('ariel', 10), bg='#E0E0E0', command=but_box)
    butt_box.place(x=30, y=485)     # Placing at appropriate location
    # Creating Button for Gaussian Filter
    butt_gauss = Button(root, text='Gaussian Filter', font=('ariel', 10), bg='#E0E0E0', command=but_gauss)
    butt_gauss.place(x=145, y=485)      # placing at appropriate location


# 12.4. Create Button Function for choosing the extent of Blurring:
def create_button_blur(filt_type):
    '''
    Takes input filter type, creates button for user to choose the extent of blurring
    and calls the Blurr Function with all the parameters.
    Also destroys the buttons for filter choices.
    '''

    # Creating buttons as global variables so that they could be destroyed using other functions
    global butt_1, butt_2, butt_3, butt_4, butt_5, butt_6, butt_7, butt_8, butt_9, butt_10, label_extent        # butt1-10: buttons for values 1 to 10, label_extent: label for asking user the extent of blurring
    # taking buttons and labels for filter choice as global variables and destroying them
    global butt_box, butt_gauss, label_filter       # variables are defined in choose_filter function
    butt_box.after(10, butt_box.destroy)        # destroy butt_box button after 10 ms
    butt_gauss.after(10, butt_gauss.destroy)        # destroy butt_gauss button after 10 ms
    label_filter.after(10, label_filter.destroy)        # adestroy label_filter label
    # Create Labels for asking user for Extent of Blurring
    label_extent = Label(root, text='Choose Extent of Blurring:', font=('ariel', 10), bg='#E0E0E0')
    label_extent.place(x=30, y=475)     # placing the label_extent at appropriate location

    # Creating functions of Buttons to call Blurr Function with diferrent paramters
    def but_1():        # For extent=1
        Blurr(1, filt_type)

    def but_2():        # For extent =2
        Blurr(2, filt_type)

    def but_3():        # For extent =3
        Blurr(3, filt_type)

    def but_4():        # For extent =4
        Blurr(4, filt_type)

    def but_5():        # For extent =5
        Blurr(5, filt_type)

    def but_6():        # For extent =6
        Blurr(6, filt_type)

    def but_7():        # For extent =7
        Blurr(7, filt_type)

    def but_8():        # For extent =8
        Blurr(8, filt_type)

    def but_9():        # For extent =9
        Blurr(9, filt_type)

    def but_10():       # For extent =10
        Blurr(10, filt_type)

    # Creating buttons for different extent of blurring
    butt_1 = Button(root, text='1', font=('ariel', 10), bg='#E0E0E0', command=but_1)        # Button for 1
    butt_1.place(x=30, y=505)
    butt_2 = Button(root, text='2', font=('ariel', 10), bg='#E0E0E0', command=but_2)        # Button for 2
    butt_2.place(x=55, y=505)
    butt_3 = Button(root, text='3', font=('ariel', 10), bg='#E0E0E0', command=but_3)        # Button for 3
    butt_3.place(x=80, y=505)
    butt_4 = Button(root, text='4', font=('ariel', 10), bg='#E0E0E0', command=but_4)        # Button for 4
    butt_4.place(x=105, y=505)
    butt_5 = Button(root, text='5', font=('ariel', 10), bg='#E0E0E0', command=but_5)        # Button for 5
    butt_5.place(x=130, y=505)
    butt_6 = Button(root, text='6', font=('ariel', 10), bg='#E0E0E0', command=but_6)        # Button for 6
    butt_6.place(x=155, y=505)
    butt_7 = Button(root, text='7', font=('ariel', 10), bg='#E0E0E0', command=but_7)        # Button for 7
    butt_7.place(x=180, y=505)
    butt_8 = Button(root, text='8', font=('ariel', 10), bg='#E0E0E0', command=but_8)        # Button for 8
    butt_8.place(x=205, y=505)
    butt_9 = Button(root, text='9', font=('ariel', 10), bg='#E0E0E0', command=but_9)        # Button for 9
    butt_9.place(x=230, y=505)
    butt_10 = Button(root, text='10', font=('ariel', 10), bg='#E0E0E0', command=but_10)     # Button for 10
    butt_10.place(x=255, y=505)


# 12.5. Function for Blurring:
def Blurr(c_ext, filt_type):
    '''
    Takes extent of blur (int) and type of the filter as the input and applies
    Blurring operation to the last modified Image
    '''

    # Global Variable stack to get the last modified image
    global stack
    # Copying the c_ext to extent before destroying the buttons
    extent = c_ext
    # Calling the delete_button() function that deletes the buttons created for choosing the extent
    delete_button()
    # Setting window_size as odd numbers depending upon the chosen extent
    window_size = extent * 2 + 1
    # Creating type of Filter: Box/ Gaussian based on user's preference
    if (filt_type == 'box'):
        filter = box_filter(window_size)        # Calling the box_filter Function and saving the filter into filter variable
    else:
        sigma = 0.8 + extent * .2       # Initializing the value of sigma depending upon extent of blurring
        filter = gaussian_filter(window_size, sigma)        # Calling the gaussian_filter Function and saving the filter into filter variable

    # Repeating the steps from Histogramm function to get the V component of last modified Image
    pix = np.copy(stack[-1])
    if (im_type == "color"):
        V = np.rollaxis(pix, -1)[2]
    else:
        V = pix
    # Padding the image
    V1 = np.pad(V, (window_size - 1, window_size - 1), 'reflect')
    # Convolving padded image with chosen filter
    G_xy = convo(V1, filter)
    # Slicing the convolution output to get the output as same shape of input
    G_xy = G_xy[window_size // 2:G_xy.shape[0] - window_size // 2, window_size // 2:G_xy.shape[1] - window_size // 2]
    G_xy = np.round(G_xy)       # Rounding to the nearest integer

    # Repeating the steps from Histogram function to store and display the image
    if (im_type == "color"):
        pix[:, :, 2] = G_xy
    else:
        pix = G_xy
    stack.append(pix)
    Display(pix)


# 13. Delete Button function for deleting the buttons:
def delete_button():
    '''
    Deletes the Buttons created for choosing extent of blurr or sharpen
    '''

    # Global variables of all the buttons and labels to destroy them
    global butt_1, butt_2, butt_3, butt_4, butt_5, butt_6, butt_7, butt_8, butt_9, butt_10, label_extent
    # Destroy all the buttons and labels after use
    butt_1.after(100, butt_1.destroy)
    butt_2.after(100, butt_2.destroy)
    butt_3.after(100, butt_3.destroy)
    butt_4.after(100, butt_4.destroy)
    butt_5.after(100, butt_5.destroy)
    butt_6.after(100, butt_6.destroy)
    butt_7.after(100, butt_7.destroy)
    butt_8.after(100, butt_8.destroy)
    butt_9.after(100, butt_9.destroy)
    butt_10.after(100, butt_10.destroy)
    label_extent.after(100, label_extent.destroy)


# 14. Function for Edge Detection- We will detect edges using sobel operator
def edge_det(method='sobel'):
    '''
    takes the edge detection method {'sobel', 'laplace1','laplace2', 'subnet'} as input and
    reurns and displays the edge detected image.
    'sobel' - for sobel operator
    'laplace1'- For laplacian first orfer
    'laplace2'- For laplacian second order
    'subnet'- For Subnet mask
    '''

    # Repeating the steps from Histogramm function to get the V component of last modified Image
    global stack
    pix = np.copy(stack[-1])

    if (im_type == "color"):
        V = pix[:, :, 2]
    else:
        V = pix

    # We will use only 3x3 window size for all the methods of edge detection
    window_size = 3
    # Padding the image to get same size as input after convolution
    V1 = np.pad(V, (window_size - 1, window_size - 1), 'reflect')
    # convolving with different filters depending upon the type of filter
    if (method == 'laplace1'):
        # Laplacian First order filter
        filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        # Convolution
        G = convo(V1, filter)
        # Slicing to get original shape back
        G = G[window_size // 2:G.shape[0] - window_size // 2, window_size // 2:G.shape[1] - window_size // 2]

    elif (method == 'laplace2'):
        # Laplacian Second order filter and same steps as laplacian second order
        filter = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        G = convo(V1, filter)
        G = G[window_size // 2:G.shape[0] - window_size // 2, window_size // 2:G.shape[1] - window_size // 2]

    elif (method == 'sobel'):
        # Sobel x and y operators
        filter_sob_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        filter_sob_y = filter_sob_x.T
        # Convolution with both operators
        Gx = convo(V1, filter_sob_x)
        Gy = convo(V1, filter_sob_y)
        # Calculating G as G=square_root(G_x^2 + G_y^2)
        G = np.sqrt(Gx ** 2 + Gy ** 2)
        # Slicing for getting original shape back
        G = G[window_size // 2:G.shape[0] - window_size // 2, window_size // 2:G.shape[1] - window_size // 2]

    elif (method == 'subnet'):
        # Using Gaussian blurr to find the subnet mask
        filter_gauss = gaussian_filter(window_size, sigma=0.8)
        # Convolution with Filter
        blurr = convo(V1, filter_gauss)
        # Slicing for getting original shape back
        blurr = blurr[window_size // 2:blurr.shape[0] - window_size // 2,
                window_size // 2:blurr.shape[1] - window_size // 2]
        # Creating the mask by taking the difference between original and blurred image
        G = V - blurr

    # Repeating the steps from Histogram function to store and display the image
    imag1 = Image.fromarray(G)
    imag1.thumbnail((1230, 530))
    imag2 = ImageTk.PhotoImage(imag1)
    canva.create_image(580, 325, image=imag2)
    canva.image = imag2

    return G        # returning G: Edge detected image for further processing into other functions


# 15. Sharpening:
# 15.1. Choosing a Method:
def choose_method():
    '''
    Makes Buttons for choosing the method and calls create_button_sharpen function
    with parameter method type
    '''

    # Making buttons and labels global variables so that they can be destroyed later
    global butt_laplace_1, butt_laplace_2, butt_sobel, butt_subnet_mask, label_method
    # Creating Label to ask user for method  (Almost same program as choosing the blurring filter: choose_filter())
    label_method = Label(root, text='Please Choose a method for Sharpening: ', font=('ariel', 10), bg="#E0E0E0")
    label_method.place(x=30, y=455)

    # Defining functions for Buttons
    # Function for sharpening using Laplacian first order operator
    def but_laplace1():
        create_button_sharpen('laplace1')

    # Function for sharpening using Laplacian second order operator
    def but_laplace2():
        create_button_sharpen('laplace2')

    # Function for sharpening using Subnet Masking method
    def but_subnet():
        create_button_sharpen('subnet')

    # Function for sharpening using sobel operator
    def but_sobel():
        create_button_sharpen('sobel')

    # Creating button for laplacian first order
    butt_laplace_1 = Button(root, text='Lapacian 1st order ', font=('ariel', 10), bg='#E0E0E0', command=but_laplace1)
    butt_laplace_1.place(x=30, y=485)
    # Creating button for subnet mask
    butt_subnet_mask = Button(root, text='Subnet Mask', font=('ariel', 10), bg='#E0E0E0', command=but_subnet, padx=6)
    butt_subnet_mask.place(x=175, y=485)
    # Creating button for laplacian second order
    butt_laplace_2 = Button(root, text='Laplacian 2nd order', font=('ariel', 10), bg='#E0E0E0', command=but_laplace2)
    butt_laplace_2.place(x=30, y=520)
    # Creating button for sobel operator
    butt_sobel = Button(root, text='Sobel Operator', font=('ariel', 10), bg='#E0E0E0', command=but_sobel)
    butt_sobel.place(x=175, y=520)


# 15.2. Buttons For choosing extent of Sharpening:
def create_button_sharpen(method):
    '''
    Takes input method to be used for sharpening, creates button for user to choose the extent of sharpening
    and calls the sharpenn Function with all the parameters.
    Also destroys the buttons for method choices.
    '''

    # Creating buttons as global variables so that they could be destroyed using other functions
    global butt_1, butt_2, butt_3, butt_4, butt_5, butt_6, butt_7, butt_8, butt_9, butt_10, label_extent
    # taking buttons and labels for method choice as global variables and destroying them
    global butt_laplace_1, butt_laplace_2, butt_sobel, butt_subnet_mask, label_method
    # Destroy the buttons and labels after use
    butt_laplace_1.after(10, butt_laplace_1.destroy)
    butt_laplace_2.after(10, butt_laplace_2.destroy)
    butt_sobel.after(10, butt_sobel.destroy)
    butt_subnet_mask.after(10, butt_subnet_mask.destroy)
    label_method.after(10, label_method.destroy)

    # Create Labels for asking Extent of Blurring by user (Almost same program as choosing the extent of blurring: create_button_blur())
    label_extent = Label(root, text='Choose Extent of Sharpening:', font=('ariel', 10), bg='#E0E0E0')
    label_extent.place(x=30, y=475)

    # Creating functions of Buttons to call sharpenn Function with diferrent paramters
    def but_1():
        sharpenn(1, method)

    def but_2():
        sharpenn(2, method)

    def but_3():
        sharpenn(3, method)

    def but_4():
        sharpenn(4, method)

    def but_5():
        sharpenn(5, method)

    def but_6():
        sharpenn(6, method)

    def but_7():
        sharpenn(7, method)

    def but_8():
        sharpenn(8, method)

    def but_9():
        sharpenn(9, method)

    def but_10():
        sharpenn(10, method)

    # Creating buttons for different extent of blurring (1:10)
    butt_1 = Button(root, text='1', font=('ariel', 10), bg='#E0E0E0', command=but_1)
    butt_1.place(x=30, y=505)
    butt_2 = Button(root, text='2', font=('ariel', 10), bg='#E0E0E0', command=but_2)
    butt_2.place(x=55, y=505)
    butt_3 = Button(root, text='3', font=('ariel', 10), bg='#E0E0E0', command=but_3)
    butt_3.place(x=80, y=505)
    butt_4 = Button(root, text='4', font=('ariel', 10), bg='#E0E0E0', command=but_4)
    butt_4.place(x=105, y=505)
    butt_5 = Button(root, text='5', font=('ariel', 10), bg='#E0E0E0', command=but_5)
    butt_5.place(x=130, y=505)
    butt_6 = Button(root, text='6', font=('ariel', 10), bg='#E0E0E0', command=but_6)
    butt_6.place(x=155, y=505)
    butt_7 = Button(root, text='7', font=('ariel', 10), bg='#E0E0E0', command=but_7)
    butt_7.place(x=180, y=505)
    butt_8 = Button(root, text='8', font=('ariel', 10), bg='#E0E0E0', command=but_8)
    butt_8.place(x=205, y=505)
    butt_9 = Button(root, text='9', font=('ariel', 10), bg='#E0E0E0', command=but_9)
    butt_9.place(x=230, y=505)
    butt_10 = Button(root, text='10', font=('ariel', 10), bg='#E0E0E0', command=but_10)
    butt_10.place(x=255, y=505)


# 15.3. Function for Sharpening:
def sharpenn(c_ext, method):
    '''
    Takes extent of blur (int) and type of the filter as the input and applies
    Blurring operation to the last modified Image
    '''
    # Global Variable stack to get the last modified image
    global stack
    # Copying the c_ext to extent before destroying the buttons
    extent = c_ext
    # Calling the delete_button() function that deletes the buttons created for choosing the extent
    delete_button()

    # Repeating the steps from Histogramm function to get the V component of last modified Image
    pix = np.copy(stack[-1])
    if (im_type == "color"):
        V = np.rollaxis(pix, -1)[2]
    else:
        V = pix
    orig_shape = V.shape

    # Calling edge detection function with the method parameter chosen
    # by user to get the mask for sharpening the image
    mask = edge_det(method)

    # Using the chosen extent by user, creating the image by adding the mask
    # with different weights
    G_xy = V * (1 - extent * .005) + mask * (extent * 0.005 + .05)
    G_xy = np.round(G_xy)

    # Repeating the steps from Histogram function to store and display the image
    if (im_type == "color"):
        pix[:, :, 2] = G_xy
    else:
        pix = G_xy
    stack.append(pix)
    Display(pix)


# 16. Local Histogram Equalization:
def L_histogram():
    '''
    Applies Local Histogram Eqalization to the last modified Image
    '''
    # Repeating the steps from Histogramm function to get the V component of last modified Image
    global stack, im_type
    pix = np.copy(stack[-1])
    if (im_type == "color"):
        V = np.rollaxis(pix, -1)[2]
    else:
        V = pix
    # Copying the original image for future use
    orig_im = np.copy(V)

    # Defining the number of pixels for each local box
    M = 12
    # N is calculated to be in same aspect ratio as the image
    N = round(M * V.shape[1] / V.shape[0])
    # Setting m and n as zeros and running a loop over image
    m = n = 0
    while (m <= V.shape[0]):
        # Making n=0 for each m
        n = 0
        while (n <= V.shape[1]):
            # Using the same Histogram Equalization technique in boxes of m x n
            hist = []
            s = {}
            # Checking if m and n are not out of the image
            if ((m + M <= V.shape[0]) and (n + N <= V.shape[1])):
                Z = V[m:m + M, n:n + N]     # Z is used to store a slice of image array
            elif (m + 1 < V.shape[0]) and (n + 1 < V.shape[1]):
                Z = V[m:, n:]
            # count number of pixels for each intensity level
            for i in range(0, 256):
                hist.append(np.count_nonzero(Z == i) / (Z.shape[0] * Z.shape[1]))
                if (i > 0):
                    s[i] = (s[i - 1] + hist[i])
                else:
                    s[i] = (hist[i])
            # Updating all the pixels with intensity r to intensity s
            for i in range(0, 256):
                s[i] = round(s[i] * 255)
                Z = np.where(Z == i, s[i], Z)
            # Updating the small part of image
            if ((m + M <= V.shape[0]) and (n + N <= V.shape[1])):
                V[m:m + M, n:n + N] = Z
            elif (m + 1 < V.shape[0]) and (n + 1 < V.shape[1]):
                V[m:, n:] = Z
            # Increament m and for moving to next box
            n = n + N
        m = m + M
    # Adding the original image and local histogram equalized image in 80:20 ratio
    V = orig_im * .80 + .20 * V
    # Repeating the steps from Histogram function to store and display the image
    if (im_type == "color"):
        pix[:, :, 2] = V
    else:
        pix = V
    stack.append(pix)
    Display(stack[-1])


# General Operation
label_general = Label(root, text='General Operations:', font=('ariel', 17), bg='#8676AD')       # Creating the label for general operations
label_general.place(x=5, y=35)      # Placing the label_general at appropriate location

# Button for Load_Image, Save, Undo, Reset and Exit
load = Button(root, text="Load", bg="#C0C0C0", font=('ariel', 11), command=selects)     # Button for load operation
load.place(x=30, y=80)      # Placing at appropriate location
# Save:
save = Button(root, text="Save", bg="#C0C0C0", font=("ariel", 11), command=Savee)       # Button for save operation
save.place(x=100, y=80)     # Placing at appropriate location
# Undo:
undo = Button(root, text="Undo", bg="#C0C0C0", font=("ariel", 11), command=Undoo)       # Button for undo operation
undo.place(x=170, y=80)     # Placing at appropriate location
# Reset:
reset = Button(root, text="Reset", bg="#C0C0C0", font=("ariel", 11), command=Resett)        # button for reset operation
reset.place(x=240, y=80)        # Placing at appropriate location
# Exit: We will use Root.distroy to exit from GUI
exxit = Button(root, text="Exit", bg="#C0C0C0", font=("ariel", 11), command=root.destroy)       # Button for exit operation
exxit.place(x=1450, y=735)      # Placing at appropriate location

# Label for Editing Options:
label_edit = Label(root, text='Editing Operations:', font=('ariel', 17), bg='#8676AD')      # Creating the label for general operations
label_edit.place(x=5, y=120)        # Placing at appropriate location

# Button for Histogram Equalization:
hist = Button(root, text="Equalize Histogram", bg='#C0C0C0', font=("ariel", 11), command=Histogramm)        # Button for Histogram Equalization
hist.place(x=80, y=165)     # Placing at appropriate location

# Button for Gamma Correction:
gamm = Button(root, text="Gamma Correction", bg='#C0C0C0', font=("ariel", 11), padx=2, command=Gammaa)
gamm.place(x=80, y=205)     # Placing at appropriate location

# Button for Local Histogram Equalization:
L_hist = Button(root, text="Local Histogram", bg='#C0C0C0', font=("ariel", 11), padx=12, command=L_histogram)
L_hist.place(x=80, y=245)       # Placing at appropriate location

# Button for Edge Detection:
egde = Button(root, text="Edge Detection", bg='#C0C0C0', font=("ariel", 11), padx=14, command=edge_det)
egde.place(x=80, y=285)     # Placing at appropriate location

# Button for Log Transform:
llog = Button(root, text="Log Transform", bg='#C0C0C0', font=("ariel", 11), padx=16.5, command=Llog)
llog.place(x=80, y=325)     # Placing at appropriate location

# Button for Sharpening:
sharpen = Button(root, text="Sharpen", bg='#C0C0C0', font=("ariel", 11), padx=37, command=choose_method)
sharpen.place(x=80, y=365)      # Placing at appropriate location

# Button for Blurring:
blur = Button(root, text="Blur", bg='#C0C0C0', font=("ariel", 11), padx=52, command=choose_filter)
blur.place(x=80, y=405)     # Placing at appropriate location

##Image Display area using Canvas:
# Main Area:
canva = Canvas(root, width="1140", height="640", relief=RIDGE, bd=5, bg='#AB9FC6')      # Creating canvas object as canva for main canvas
canva.place(x=325, y=50)        # Placing at appropriate location
# Small Area:
canva2 = Canvas(root, width="228", height="128", relief=RIDGE, bd=5, bg='#AB9FC6')      # Creating canva2 as canvas object for small canvas
canva2.place(x=30, y=562)       # Placing at appropriate location

# MainLoop:
root.mainloop()     # mainloop for continuously running the GUI

