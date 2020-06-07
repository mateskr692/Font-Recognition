import trdg.generators as gens
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from PIL import Image
import os


class data_generator:

    def GenerareDataset(self, fontsDir: str = r".venv\Lib\site-packages\trdg\fonts\latin",  outDir: str = r"Dataset") -> None:

        os.mkdir(os.path.join(outDir, "Train"))
        os.mkdir(os.path.join(outDir, "Test"))
        for filename in os.listdir(fontsDir):

            fontName = os.path.splitext(filename)[0]
            fontFullPath = os.path.join(fontsDir, filename)

            fontTrainDir = os.path.join(outDir, "Train", fontName)
            fontTestDir = os.path.join(outDir, "Test", fontName)
            os.mkdir(fontTestDir)
            os.mkdir(fontTrainDir)

            imgIndex = 0

            def generate(path: str, generator: gens.GeneratorFromDict) -> None:
                nonlocal imgIndex
                for [img, _] in generator:
                    img.save(os.path.join(path, str(imgIndex) + ".jpg"))
                    imgIndex += 1
                    break
                return None

            # generate data :
            generator = gens.GeneratorFromDict(
                fonts=[fontFullPath],
                random_blur=True,
                random_skew=True,
                blur=1,
                skewing_angle=3,
                image_dir="Backgrounds",
            )

            # Train Data
            # colored text on noisy background
            for _ in range(0, 200):
                color = '#' + \
                    str(format(random.randint(20, 160), 'x')) + \
                    str(format(random.randint(20, 160), 'x')) + \
                    str(format(random.randint(20, 160), 'x'))

                generator.generator.text_color = color
                generator.generator.background_type = random.randint(0, 2)
                generate(fontTrainDir, generator)

            # colored text on background
            generator.generator.random_blur = False
            generator.generator.blur = 0
            generator.generator.background_type = 3
            for _ in range(0, 50):
                color = '#' + \
                    str(format(random.randint(20, 80), 'x')) + \
                    str(format(random.randint(20, 80), 'x')) + \
                    str(format(random.randint(20, 80), 'x'))

                generator.generator.text_color = color
                generate(fontTrainDir, generator)

            # Test Data
            for _ in range(0, 20):
                color = '#' + \
                    str(format(random.randint(20, 160), 'x')) + \
                    str(format(random.randint(20, 160), 'x')) + \
                    str(format(random.randint(20, 160), 'x'))

                generator.generator.text_color = color
                generator.generator.background_type = random.randint(0, 2)
                generate(fontTestDir, generator)

            # colored text on background
            generator.generator.random_blur = False
            generator.generator.blur = 0
            generator.generator.background_type = 3
            for _ in range(0, 5):
                color = '#' + \
                    str(format(random.randint(20, 80), 'x')) + \
                    str(format(random.randint(20, 80), 'x')) + \
                    str(format(random.randint(20, 80), 'x'))

                generator.generator.text_color = color
                generate(fontTestDir, generator)

        return None

    def GenerateRandomImage(self, text: str = "") -> Image:
        if(text == ""):
            generator = gens.GeneratorFromDict()
            for img, _ in generator:
                return img
        else:
            generator = gens.GeneratorFromWikipedia()
            for img, _ in generator:
                return img
