from evaluate import Evaluator
from models import load_model
from visualizations import Visualizer, save_video


def main():
    model = load_model("supertuxcart_cnn", with_weights=True)

    # run the model on the track
    visualizer = Visualizer()

    # set visualizer to None if you don't want to visualize
    evaluator = Evaluator(model, visualizer=visualizer)
    evaluator.evaluate(track_name='lighthouse', max_steps=1500)

    # list of images (numpy array)
    frames = visualizer.frames

    # visualize however you like (plt.imshow, make a video, etc)
    save_video(frames, "video.mp4")


if __name__ == "__main__":
    main()