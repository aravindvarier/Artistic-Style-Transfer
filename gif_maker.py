import imageio

gif_file = './image_saves/shruti_theshipwreckoftheminotaur/theshipwreckoftheminotaur.gif'
movie_file = './image_saves/shruti_theshipwreckoftheminotaur/theshipwreckoftheminotaur.mp4'

with imageio.get_writer(gif_file, mode='I') as writer:
    for i in range(500, 70500, 500):
        image = imageio.imread("image_saves/shruti_theshipwreckoftheminotaur/" + str(i) + ".jpg")
        writer.append_data(image)


