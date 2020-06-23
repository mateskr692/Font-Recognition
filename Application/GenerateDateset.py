import trdg.generators as gens
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import os



class data_generator:

    fontsDir: str
    outDir: str

    def __init__(self, fontsDir: str = r"Fonts",  outDir: str = r"Dataset", bgDir: str = r"Backgrounds"):
        self.fontsDir = fontsDir
        self.outDir = outDir
        self.bgDir = bgDir

    def GenerateBackgrounds(self) -> None:
        for i in range(160, 256, 5):
            img = Image.new('RGB', (500, 500), (i, i, i))
            img.save(os.path.join(self.bgDir, "bg_" + str(i) + ".png"))

        return None

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
                blur=2,
                image_dir=self.bgDir,
                fit=True,
            )

            # Dictionary generator
            for _ in range(0, 100):
                size = random.randint(30, 80)
                bg_type = 3  # random.randint(0, 2)  # 3 to include images
                brightness = '{:02x}'.format(random.randint(0, 50))
                spacing = random.randint(0, 5)

                generator.generator.size = size
                generator.generator.background_type = bg_type
                generator.generator.text_color = "#" + brightness*3
                generator.generator.character_spacing = spacing

                def generate(path: str, amount: int) -> None:
                    nonlocal imgIndex
                    nonlocal generator
                    nonlocal size
                    nonlocal bg_type
                    indx = 0
                    for [img, _] in generator:
                        img = img.convert('L')
                        if img.width > img.height:
                            beg = random.randint(0, img.width-img.height - 1)
                            img = img.crop((beg, 0, img.height+beg, img.height))
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

            generator = gens.GeneratorFromRandom(
                fonts=[fontFullPath],
                random_blur=True,
                blur=2,
                image_dir=self.bgDir,
                fit=True,
                length=3
            )

            # Random generator
            for _ in range(0, 100):
                size = random.randint(30, 80)
                bg_type = 3  # random.randint(0, 2)  # 3 to include images
                brightness = '{:02x}'.format(random.randint(0, 50))
                spacing = random.randint(0, 5)

                generator.generator.size = size
                generator.generator.background_type = bg_type
                generator.generator.text_color = "#" + brightness*3
                generator.generator.character_spacing = spacing

                generate(fontTestDir, 1)
                generate(fontTrainDir, 10)

        return None


# g = gens.GeneratorFromRandom(
#     image_dir="Backgrounds",
#     background_type=3,
#     size=64,
#     fit=True,
#     character_spacing=0,
#     length=3,
# )
# for i, _ in g:
#     i.show()
#     break


gen = data_generator()
gen.GenerareDataset()
