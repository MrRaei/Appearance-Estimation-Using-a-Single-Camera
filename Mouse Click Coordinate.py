import Tkinter
from PIL import Image, ImageTk
from sys import argv

window = Tkinter.Tk(className="Img")

image = Image.open(argv[1] if len(argv) >=2 else "uni/vid/amol/Amol_calibre2.jpg")
canvas = Tkinter.Canvas(window, width=image.size[0], height=image.size[1])
canvas.pack()
image_tk = ImageTk.PhotoImage(image)
canvas.create_image(image.size[0]//2, image.size[1]//2, image=image_tk)

def callback(event):
    print "[[",float(event.x), ",", float(event.y),"]] ,",
    #print float(event.x), ",", float(event.y)

canvas.bind("<Button-1>", callback)
Tkinter.mainloop()


# [[ 841.0 , 26.0 ]]
# [[ 859.0 , 26.0 ]]	+18
# [[ 894.0 , 26.0 ]]	+35		+53



# [[ 317.0 , 520.0 ]]				[[ 383.0 , 463.0 ]]
# [[ 331.0 , 516.0 ]]	14,-4		[[ 396.0 , 460.0 ]]		13 , -3
# [[ 359.0 , 508.0 ]]	28,-8		[[ 424.0 , 454.0 ]]		28 , -6		41 , -9