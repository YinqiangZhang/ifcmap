import os
import glob
import tqdm
import imageio

root_path = os.path.dirname(os.path.abspath(__file__))
filenames = glob.glob(os.path.join(root_path, 'video_figs', '*.png'))
with imageio.get_writer(os.path.join(root_path, 'registration.gif'), mode='I', fps=25) as writer:
    for idx, filename in tqdm.tqdm(enumerate(filenames), leave=False):
        if idx % 2 == 0:
            image = imageio.imread(filename)
            writer.append_data(image)