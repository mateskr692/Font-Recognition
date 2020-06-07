import trdg.generators as gens
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import os


def GenerareDataset() -> None:

    fontsDir = r".venv\Lib\site-packages\trdg\fonts\latin"
    outDir = r"Dataset"
    fontsDictionary = open(outDir + r"\fonts.txt", "w")

    for filename in os.listdir(fontsDir):

        # create directory and add fontname to dictionary
        fontName = os.path.splitext(filename)[0]
        fontFullPath = fontsDir + "\\" + filename
        fontImageDirectory = outDir + "\\" + fontName
        os.mkdir(fontImageDirectory)
        fontsDictionary.write(fontName + "\n")

        imgIndex = 0

        def generate(count: int, generator: gens.GeneratorFromWikipedia) -> None:
            i = 0
            for img, _ in generator:
                img.save(fontImageDirectory + "\\" + str(imgIndex) + ".jpg")
                i += 1
                imgIndex += 1
                if i > count:
                    break

            return None

        # generate data
        generator = gens.GeneratorFromWikipedia(
            fonts=[fontFullPath],
            random_blur=True,
            random_skew=True,
            blur=1,
            skewing_angle=3,
        )
        generate(20, generator)

        break

    return None


def GenerateRandomImage(text: str = "") -> Image:
    if(text == ""):
        generator = gens.GeneratorFromDict()
        for img, _ in generator:
            return img
    else:
        generator = gens.GeneratorFromWikipedia()
        for img, _ in generator:
            return img
