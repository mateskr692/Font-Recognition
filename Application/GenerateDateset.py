import trdg.generators as gens
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from PIL import Image
import os



class data_generator:

    fontsDir: str
    outDir: str

    def __init__(self, fontsDir: str = r"Fonts",  outDir: str = r"Dataset"):
        self.fontsDir = fontsDir
        self.outDir = outDir

    def GenerareDataset(self) -> None:

        os.mkdir(os.path.join(self.outDir, "Train"))
        os.mkdir(os.path.join(self.outDir, "Test"))
        for filename in os.listdir(self.fontsDir):

            fontName = os.path.splitext(filename)[0]
            fontFullPath = os.path.join(self.fontsDir, filename)

            fontTrainDir = os.path.join(self.outDir, "Train", fontName)
            fontTestDir = os.path.join(self.outDir, "Test", fontName)
            os.mkdir(fontTestDir)
            os.mkdir(fontTrainDir)

            imgIndex = 0
            generator = gens.GeneratorFromDict(
                fonts=[fontFullPath],
                random_blur=True,
                random_skew=True,
                blur=1,
                skewing_angle=3,
                image_dir="Backgrounds",
                size=64,
            )

            for _ in range(0, 200):
                size = random.randint(32, 80)
                bg_type = random.randint(0, 3)
                brightness = '{:02x}'.format(random.randint(0, 50))

                generator.generator.size = size
                generator.generator.background_type = bg_type
                generator.generator.text_color = "#" + brightness*3
                generator.generator.margins = (random.randint(2, 10),
                                               random.randint(2, 10),
                                               random.randint(2, 10),
                                               random.randint(2, 10))

                def generate(path: str, amount: int) -> None:
                    nonlocal imgIndex
                    nonlocal generator
                    nonlocal size
                    nonlocal bg_type
                    indx = 0
                    for [img, _] in generator:
                        img = img.convert('L')
                        img.save(os.path.join(
                            path, str(imgIndex) + "_size" + str(size) +
                            "_bgtype" + str(bg_type) + ".jpg"))
                        imgIndex += 1
                        indx += 1
                        if indx >= amount:
                            break
                    return None

                generate(fontTestDir, 1)
                generate(fontTrainDir, 10)

        return None

    def GenerateRandomImage(self, text: str = "", font: str = "") -> Image:

        filename: str
        fontname: str

        generator = gens.GeneratorFromDict(
            random_blur=True,
            random_skew=True,
            blur=1,
            skewing_angle=3,
            image_dir="Backgrounds",
            size=64,
        )

        if(text != ""):
            generator = gens.GeneratorFromStrings(
                strings=[text],
                random_blur=True,
                random_skew=True,
                blur=1,
                skewing_angle=3,
                image_dir="Backgrounds",
                size=64,
            )

        if(font != ""):
            if(os.path.isfile(font)):
                filename = font
                fontname = os.path.splitext(filename)[0]
                generator.fonts = [filename]

            else:
                print("Invalid font, picking one at random")

        filename = random.choice(os.listdir(self.fontsDir))
        fontname = os.path.splitext(filename)[0]
        generator.fonts = [os.path.join(self.fontsDir, filename)]


# g = gens.GeneratorFromDict(
#     random_blur=False,
#     random_skew=False,
#     blur=0,
#     skewing_angle=0,
#     image_dir="Backgrounds",
#     size=64,
#     fit=True,
#     margins=(random.randint(0, 10), random.randint(0, 10),
#              random.randint(0, 10), random.randint(0, 10))
# )
# for i, _ in g:
#     i.save("Test.jpg")
#     break
gen = data_generator()
gen.GenerareDataset()
