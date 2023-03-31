import math
from manim import *
from PIL import Image


class PixelGrid(Scene):
    def construct(self):
        # Load image and convert to numpy array
        self.wait(1)
        corona= ImageMobject("download.jpeg")
        corona.scale(3.0)
        # corona.center()

        # self.add(corona)
        self.wait(1)


        image = Image.open("download.jpeg")
        image_data = np.array(image)

        # Define parameters
        block_size = 10
        block_row = math.ceil(image_data.shape[0]/block_size)
        block_col = math.ceil(image_data.shape[1]/block_size)

        # Split image into blocks
        blocks = [image_data[i:i+block_size, j:j+block_size] for i in range(0, image_data.shape[0], block_size) for j in range(0, image_data.shape[1], block_size)]

        # Create grid of blocks
        self.add(corona)
        self.wait(1)
        firstBlock = ImageMobject(blocks[0])
        firstBlock.scale(3)
        # print(corona.get_corner(UL))
        firstBlock.move_to(corona.get_corner(UL) + np.array([firstBlock.width, -firstBlock.height, 0])/2)
        
        # Create a surrounding rectangle
        rect = SurroundingRectangle(firstBlock, buff=0.05)

        # Highlight the circle by showing the surrounding rectangle
        self.play(Create(rect))
        self.wait(1)

        # Unhighlight the circle by removing the surrounding rectangle
        self.play(FadeOut(rect))
        self.add(firstBlock)
        self.wait(1)

        self.remove(corona)
        self.wait(1)

        expand_anim = Transform(firstBlock, firstBlock.copy().scale(10).center(), run_time=1)
        self.play(expand_anim)

        self.wait(1)

        rgb_text = MathTex("\\begin{pmatrix} 1 & 2 & 3 & 4 & 5 \\\\ 1 & 2 & 3 & 4 & 5 \\\\1 & 2 & 3 & 4 & 5 \\\\1 & 2 & 3 & 4 & 5 \\\\1 & 2 & 3 & 4 & 5 \\\\\\end{pmatrix}").scale(1)
        self.add(rgb_text)
        self.play(AnimationGroup(rgb_text.animate.shift(RIGHT*3), firstBlock.animate.shift(LEFT*3)))

        equalsTex = MathTex("=")
        self.add(equalsTex)
        self.wait(2)

        self.remove(firstBlock)
        self.remove(equalsTex)
        self.play(AnimationGroup(rgb_text.animate.shift(LEFT*3)))

        dcttext = Text("Discrete Cosine Transform").scale(0.5)
        self.add(dcttext)
        self.play(dcttext.animate.shift(UP*3))
        self.wait(2)

        self.play(FadeOut(rgb_text))
        rgb_text = MathTex("\\begin{pmatrix} 1 & 2 & 3 & 4 & 5 \\\\ 1 & 2 & 3 & 4 & 0 \\\\1 & 1 & 1 & 0 & 0 \\\\1 & 1 & 0 & 0 & 0 \\\\1 & 0 & 0 & 0 & 0 \\\\\\end{pmatrix}").scale(1)
        self.add(rgb_text)
        self.wait(2)

        self.play(AnimationGroup(rgb_text.animate.shift(LEFT*3), dcttext.animate.shift(LEFT*3)))

        bittext = Text("BitMatrix").scale(0.5)
        self.add(bittext)
        self.play(bittext.animate.shift(UP*3))

        bitmatrix = MathTex("\\begin{pmatrix} 6 & 4 & 1 & 0 & 0 \\\\ 4 & 4 & 0 & 0 & 0 \\\\1 & 0 & 0 & 0 & 0 \\\\0 & 0 & 0 & 0 & 0 \\\\0 & 0 & 0 & 0 & 0 \\\\\\end{pmatrix}").scale(1)
        self.play(Write(bitmatrix))
        self.play(AnimationGroup(bitmatrix.animate.shift(RIGHT*3), bittext.animate.shift(RIGHT*3)))

        rect = SurroundingRectangle(bitmatrix[0][6], buff=0.1)
        rect.set_color(RED)
        self.play(Create(rect))

        rect2 = SurroundingRectangle(rgb_text[0][6], buff=0.1)
        rect2.set_color(YELLOW)
        self.play(Create(rect2))
        self.wait(1)

class PixelGrid2(Scene):
    def construct(self):
        # Load image and convert to numpy array
        first = MathTex("1 \\rightarrow Q \\rightarrow [1, 2^6]")
        self.add(first)
        self.wait(1)

        rect2 = SurroundingRectangle(first[0][0], buff=0.1)
        rect2.set_color(YELLOW)
        self.play(Create(rect2))
        self.wait(1)

        rect = SurroundingRectangle(first[0][8], buff=0.1)
        rect.set_color(RED)
        self.play(Create(rect))
        self.wait(1)

        qof = MathTex("Q(1) = 0b110111")
        self.clear()

        self.add(qof)
        self.wait(1)







        

