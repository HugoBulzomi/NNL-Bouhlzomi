from tkinter import *
from tkinter.ttk import *
import cv2
import csv
import json
from PIL import Image
from PIL import ImageTk
from math import sqrt
from shapely.geometry import Point, box
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
# **** Parameters ****
categories_json_file_path = "categories2.json"
target_img_size = (500, 500)
categories = []

# **** Rectangles global variables ****
start_x = 0
start_y = 0
end_x = 0
end_y = 0
current = None
rect_cpt = 0
rectangles = {} # Association type: rectangle: {"id":int, "origins":(start_x, start_x), "ends":(end_x, end_y), "label":String, "shape":Box}

current_img_name = ""
current_img_scaling = (1.0, 1.0)
cv2_img = None


# ***** Loading Related *****

def load_categories(path):
    global categories
    with open(path) as f:
        return json.load(f)["categories"]

def openImage(path, dimensions):
    global cv2_img, current_img_scaling
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    shape = list(img.shape)
    current_img_original_dimensions = (shape[0], shape[1])
    scale = 1000/max(shape[0], shape[1])
    shape[0] = min(shape[0], int(shape[0]*scale))
    shape[1] = min(shape[1], int(shape[1]*scale))
    current_img_scaling = shape[0]/current_img_original_dimensions[0]
    img = cv2.resize(img, (shape[1], shape[0]), interpolation = cv2.INTER_AREA)
    
        
    if img.dtype == "uint16":
        img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
    cv2_img = img
    extracted_name = path.split('/')[-1].split('.')
    extracted_name[-1]=".png"
    extracted_name = "ExtractedImages\\"+"".join(extracted_name)
    
    #cv2.imwrite(extracted_name, img)
    height, width, channels = img.shape
        
    # cv2 represents colors in BRG order so we need to invert those

    image_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_RGB = Image.fromarray(image_RGB)
    image_RGB = ImageTk.PhotoImage(image=image_RGB)

        
    return height, width, image_RGB

# ***** Event Related *****

def click_event(event):
    global current, start_x, start_y
    start_x = event.x
    start_y = event.y
    current = canvas.create_rectangle(start_x, start_y, start_x, start_y, fill="")

def double_click_event(event):
    global canvas
    point = Point(event.x, event.y)

    overlaps = []
    for b in rectangles.keys():
        if point.within(rectangles[b]["shape"]):
            overlaps.append(b)
    if len(overlaps) > 0:
        canvas.itemconfig(overlaps[0], outline='red')
        fire_popup(editing=overlaps[0])

def move_pressed_event(event):
    global current, start_x, start_y, end_x, end_y
    if current != None:
        end_x = event.x
        end_y = event.y
        canvas.coords(current, start_x, start_y, end_x, end_y)

def released_event(event):
    global canvas, current, start_x, start_y, end_x, end_y, categ_menu
    end_x = event.x
    end_y = event.y
    if current != None:
        if rec_area(start_x, start_y, end_x, end_y)>40 and current_img_name!="":
            canvas.coords(current, start_x, start_y, end_x, end_y)

            current_shape = box(start_x, start_y, end_x, end_y)
            for b in rectangles.keys():
                if current_shape.intersects(rectangles[b]["shape"]):
                    area = current_shape.intersection(rectangles[b]["shape"]).area
                    if max(area/current_shape.area*100, area/rectangles[b]["shape"].area*100) > 20.0:
                        canvas.delete(current)
                        current = None
                        return
                    
            fire_popup()
        else:
            canvas.delete(current)
            current = None

# ***** Popup Related *****

def fire_popup(editing=None):
    global rectangles, categories, start_x, start_y, end_x, end_y, rect_cpt
    pop = Toplevel()
    pop.geometry('350x250')
    pop.resizable(False, False)
    pop.wm_title("Enter a category for this box !")
    pop.protocol("WM_DELETE_WINDOW", root.destroy)
    pop.grab_set()

    msg = "Please, select a category for the box you created."
    if editing!=None:
        msg = "Editing box."
    
    title = Label(pop, text=msg)
    title.pack(pady=5, side=TOP)

    sep = Separator(pop, orient=HORIZONTAL)
    sep.pack(side=TOP, fill='x')

    upper_frame = Frame(pop)
    upper_frame.pack(pady=10, side=TOP)

    if editing == None:
        Label(upper_frame, text="Box Number: {}".format(rect_cpt)).grid(row=0, column=0)
        Label(upper_frame, text="Box origins: ({}, {})".format(start_x, start_y)).grid(row=1, column=0)
        Label(upper_frame, text="Box ends: ({}, {})".format(end_x, end_y)).grid(row=1, column=1)
    else:
        box_info = rectangles[editing]
        Label(upper_frame, text="Box Number: {}".format(box_info["id"])).grid(row=0, column=0)
        Label(upper_frame, text="Box origins: ({}, {})".format(box_info["origins"][0], box_info["origins"][1])).grid(row=1, column=0)
        Label(upper_frame, text="Box ends: ({}, {})".format(box_info["ends"][0], box_info["ends"][1])).grid(row=1, column=1)

    variable = StringVar(pop)
    w = OptionMenu(pop, variable, categories[0], *categories)
    w.pack(side=TOP)
    add_b = Button(pop, text ="Add category", command = lambda: fire_new_cat(pop, editing)).pack()
    add_b = Button(pop, text ="Modify category", command = lambda: fire_mod_cat(pop, editing)).pack()
    if editing!=None:
        box_info = rectangles[editing]
        variable.set(box_info["label"])
    else:
        variable.set(categories[0])

    bottom_frame = Frame(pop)
    bottom_frame.pack(pady=20,side=TOP)
    
    cancel_b = Button(bottom_frame, text ="Cancel", command = lambda: cancel_popup(pop, editing)).grid(row=0, column=0)
    ok_b = Button(bottom_frame, text ="Ok", command = lambda: validate_popup(pop, variable.get(), editing)).grid(row=0, column=1)
    if editing != None:
        del_b = Button(bottom_frame, text ="Delete", command = lambda: delete_popup(pop, editing)).grid(row=0, column=2)
    
def cancel_popup(popup, editing=None):
    global canvas, current
    if editing==None:
        canvas.delete(current)
        current = None
    else:
        canvas.itemconfig(editing, outline='black')
    popup.grab_release()
    popup.destroy()

def validate_popup(popup, label, editing=None):
    global canvas, rectangles, current, start_x, start_y, end_x, end_y, rect_cpt
    if editing == None:
        rectangles[current] = {"id":rect_cpt, "origins":(start_x, start_y), "ends":(end_x, end_y), "label":label, "shape":box(start_x, start_y, end_x, end_y)}
        rect_cpt+=1
        print(rectangles[current])
    else:
        rectangles[editing]["label"] = label
        canvas.itemconfig(editing, outline='black')
        print(rectangles[editing])

    current = None
    popup.grab_release()
    popup.destroy()

def delete_popup(popup, editing):
    global canvas, rectangles
    del rectangles[editing]
    canvas.delete(editing)
    popup.grab_release()
    popup.destroy()

def fire_new_cat(parent_popup, editing):
    pop = Toplevel()
    pop.geometry('300x70')
    pop.resizable(False, False)
    pop.wm_title("Enter a new category !")
    pop.grab_set()

    label = Label(pop, text="New category name: ").grid(row=0, column=0)
    new_name = Entry(pop)
    new_name.grid(row=0, column=1)
    ok_b = Button(pop, text ="Ok", command = lambda: add_categ(new_name.get(), pop, parent_popup, editing)).grid(row=1, column=0)
    cancel_b = Button(pop, text ="Cancel", command = lambda: close_categ_popup(pop, parent_popup)).grid(row=1, column=1)

def add_categ(name, popup, parent_popup, editing):
    global categories
    if not name in categories and name!='':
        categories.append(name)

    popup.destroy()
    parent_popup.destroy()
    fire_popup(editing)

def fire_mod_cat(parent_popup, editing):
    pop = Toplevel()
    pop.geometry('500x70')
    pop.resizable(False, False)
    pop.wm_title("Modify an existing category !")
    pop.grab_set()

    label = Label(pop, text="Old category name: ").grid(row=0, column=0)
    variable = StringVar(pop)
    w = OptionMenu(pop, variable, categories[0], *categories).grid(row=0, column=1)

    label = Label(pop, text="New category name: ").grid(row=0, column=2)
    new_name = Entry(pop)
    new_name.grid(row=0, column=3)

    check_var = IntVar()
    check = Checkbutton(pop, text = "Apply to all saved annotations", variable=check_var).grid(row=1, column=0)

    ok_b = Button(pop, text ="Ok", command = lambda: modify_categ(variable.get(), new_name.get(), pop, parent_popup, editing, check_var.get())).grid(row=2, column=0)
    cancel_b = Button(pop, text ="Cancel", command = lambda: close_categ_popup(pop, parent_popup)).grid(row=2, column=1)




def modify_categ(old, new, popup, parent_popup, editing, modify_all):
    global categories
    
    if new!='':
        for rec in rectangles.keys():
            if rectangles[rec]["label"]==old:
                rectangles[rec]["label"]=new
                
    if not (new in categories) and modify_all != 1:
        categories.append(new)
        categories = [cat if cat!=old else new for cat in categories]
    elif not (new in categories) and modify_all == 1:
        categories = [cat if cat!=old else new for cat in categories]
    elif new in categories and modify_all != 1:
        pass
    elif new in categories and modify_all == 1:
        categories.remove(old)

    print(categories)
    
    if modify_all == 1:
        with open('content.json', 'r') as infile:
            data = json.load(infile)
        for im in data.keys():
            for rec in range(len(data[im])):
                if data[im][rec]["label"] == old:
                    data[im][rec]["label"] = new
        with open('content.json', 'w') as outfile:
            json.dump(data, outfile)
    
    popup.destroy()
    parent_popup.destroy()
    fire_popup(editing)

def close_categ_popup(popup, parent_popup):
    popup.destroy()
    parent_popup.grab_set()


def onSave():
    global cv2_img
    if current_img_name != "":
        values = []
        for x in rectangles.values():
            values.append({"id":x["id"], "origins":x["origins"], "ends":x["ends"], "label":x["label"]})
            crop_start = (min(x["origins"][0], x["ends"][0]), min(x["origins"][1], x["ends"][1]))
            crop_ends = (max(x["origins"][0], x["ends"][0]), max(x["origins"][1], x["ends"][1]))
            
            crop_img = cv2_img[crop_start[1]:crop_ends[1], crop_start[0]:crop_ends[0]]
            resized = cv2.resize(crop_img, (128, 128), interpolation = cv2.INTER_AREA)
  
            #print("bounding_boxes_photos\\{}\\{}-bb-{}x{}-{}-{}.png".format(x['label'], current_img_name, x["origins"][0], x["origins"][1], x["ends"][0]-x["origins"][0], x["ends"][1]-x["origins"][1]))
            cv2.imwrite("..\\img\\bounding_boxes_photos\\{}\\{}-bb-{}x{}-{}-{}.png".format(x['label'], current_img_name, int(crop_start[0]*(1.0/current_img_scaling)), int(crop_start[1]*(1.0/current_img_scaling)), int((crop_ends[0]-crop_start[0])*(1.0/current_img_scaling)), int((crop_ends[1]-crop_start[1])*(1.0/current_img_scaling))), resized)
            #im_crop.save("bounding_boxes_photos\\{}\\{}\\-bb-{}x{}-{}-{}.png".format(x['label'], current_img_name, x["origins"][0], x["origins"][1], x["ends"][0]-x["origins"][0], x["ends"][1]-x["origins"][1]), quality=95)

        with open('content.json', 'r') as infile:
            data = json.load(infile)
        
        with open('content.json', 'w') as outfile:
            data[current_img_name] = values
            json.dump(data, outfile)
    
    

def onLoad():
    global rect_cpt, rectangles, categories, rect_cpt
    if current_img_name != "":
        with open('content.json', 'r') as infile:
            data = json.load(infile)

        values = data.get(current_img_name)
        max_id = 0
        
        if values != None:
            for rec in rectangles.keys():
                canvas.delete(rec)
                print(rec)

            rectangles = {}

            for val in values:
                new_rect = canvas.create_rectangle(val["origins"][0], val["origins"][1], val["ends"][0], val["ends"][1], fill="")
                new_box = box(val["origins"][0], val["origins"][1], val["ends"][0], val["ends"][1])
                val["shape"]= new_box
                rectangles[new_rect] = val
                max_id = max(max_id, val["id"])
                #categories.append(val["label"])
        #categories = list(set(categories))
        rect_cpt = max_id
 
        current_categs = set([rect["label"] for rect in rectangles.values()])
        inter = set(categories).intersection(current_categs)
        print(current_categs)
        print(inter)
        if inter != set(current_categs):
            answer = messagebox.askyesnocancel(title="Categories Incompatibility", message="Some categories in the annotations you are trying to load are not present in your currently loaded categories. Would you like to discard the annotations of the current image ? If not, the missing categories will be added to yours.")
            if answer == True:
                new_rectangles = {}
                for rec in rectangles.keys():
                    if not (rectangles[rec]["label"] in categories):
                        canvas.delete(rec)
                    else:
                        new_rectangles[rec] = rectangles[rec]
                rectangles = new_rectangles
            if answer == False:
                categories += list(current_categs-inter)


def loadImage():
    global canvas, img, rectangles, rect_cpt, current_img_name
    filename = filedialog.askopenfilename(initialdir = "./", title = "Select a File", filetypes = (("jpeg files","*.jpg*"), ("png files","*.png*")))
    if filename != "":
        current_img_name = ''.join(filename.split('/')[-1].split('.')[:-1])
        print(current_img_name)
        canvas.delete('all')
        rectangles = {}
        rect_cpt = 0
        height, width, img = openImage(filename, target_img_size)

        canvas.config(width=width, height=height)
        canvas.create_image(0,0, anchor=NW, image=img)


def loadCategories():
    global categories, rectangles

    filename = filedialog.askopenfilename(initialdir = "./", title = "Select a File", filetypes = (("json files","*.json*"), ("csv files","*.csv*")))
    if filename != "":
        print(filename)
        if filename[-4:] == "json":
            with open(filename, 'r') as infile:
                new_categories = json.load(infile)["categories"]
        else:
            with open(filename, 'r') as infile:
                reader = csv.reader(infile, delimiter=',', quotechar='|')
                new_categories = []
                for row in reader:
                    new_categories += [value for value in row if value != '']

        if current_img_name != "":
            current_categs = set([rect["label"] for rect in rectangles.values()])
            inter = set(new_categories).intersection(current_categs)
            if inter != set(current_categs):
                answer = messagebox.askyesnocancel(title="Categories Incompatibility", message="Some categories of the currently loaded image are missing in the file you are trying to load. Would you like to discard the annotations of the current image ? If not, the missing categories will be added to yours.")
                if answer == True:
                    categories = new_categories
                    new_rectangles = {}
                    for rec in rectangles.keys():
                        if not (rectangles[rec]["label"] in categories):
                            canvas.delete(rec)
                        else:
                            new_rectangles[rec] = rectangles[rec]
                    rectangles = new_rectangles
                if answer == False:
                    new_categories += list(current_categs-inter)
                    categories = new_categories
            else:
                categories = new_categories
        else:
            categories = new_categories

def saveCategories():
    with open('saved_categories.json', 'w') as outfile:
        json.dump({"categories":categories}, outfile)

    with open('saved_categories.csv', 'w') as outfile:
        write = csv.writer(outfile) 
        write.writerow(categories) 
                    

    
    

# ***** Utilitary Stuff *****

def rec_area(startx, starty, endx, endy):
    return abs(endx-startx)*abs(endy-starty)
    
    

# We instanciate our root tk window
root = Tk()
root.resizable(False, False)
menubar = Menu(root)
root.config(menu=menubar)

fileMenu = Menu(menubar)
fileMenu.add_command(label="Load Image", command=loadImage)
fileMenu.add_command(label="Save Annotations", command=onSave)
fileMenu.add_command(label="Load Annotations", command=onLoad)
fileMenu.add_command(label="Save Categories", command=saveCategories)
fileMenu.add_command(label="Load Categories", command=loadCategories)
menubar.add_cascade(label="File", menu=fileMenu)

# We create the canvas for the image and bind the events to our methods
canvas = Canvas(root, width = target_img_size[0], height = target_img_size[1])
canvas.bind("<ButtonPress-1>", click_event)
canvas.bind("<B1-Motion>", move_pressed_event)
canvas.bind("<ButtonRelease-1>", released_event)
canvas.bind("<Double-Button-1>", double_click_event)
canvas.pack()

# We load our categories from the json file
categories = load_categories(categories_json_file_path)

# We create our contextual menu that lets us assign a category to
# the rectangle we just drawn
'''
categ_menu = Menu(root, tearoff = 0)
for label in categories:
    categ_menu.add_command(label = label)
categ_menu.add_separator()
categ_menu.add_command(label = "Cancel")
'''

# We load an image
#img = load_image("ToAnnotate\\"+current_img_name, target_img_size)  
#canvas.create_image(0,0, anchor=NW, image=img)

mainloop()  
