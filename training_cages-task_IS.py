from kartezio.dataset import read_dataset
from skimage.data import coins
import cv2
import numpy as np
from kartezio.apps.segmentation import create_segmentation_model
from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.training import train_model
from kartezio.export import GenomeToPython
from numena.image.drawing import draw_overlay
from kartezio.fitness import FitnessAP
from kartezio.endpoint import EndpointWatershed, LocalMaxWatershed

dataset_path = "/Users/mathieu/Desktop/Kartezio/Cages-task/"

def create_model(n_inputs):
    n_iterations = 20_000
    n_children = 5
    n_nodes = 30
    model = create_instance_segmentation_model(
        n_iterations,
        n_children,
        inputs=n_inputs,
        nodes=n_nodes,
        endpoint=EndpointWatershed(markers_distance=100),
        # endpoint=LocalMaxWatershed(),
        #fitness=FitnessAP(0.5)
    )
    return model

def main():
    destination_directory = './models'
    #dataset = load_dataset(split=True)
    # dataset = read_dataset(dataset_path, preview=True)
    dataset = read_dataset(dataset_path, filename="dataset_cages-task.csv", meta_filename="META_cages-task.JSON", preview=True)
    train_x, train_y, train_v = dataset.train_xyv
    # for i in range(len(train_x)):
    #     print(train_v[i].shape, train_y[i][0].shape)
    #     cv2.imwrite(f'coins{i}.png', np.hstack((train_v[i], train_y[i][0])))

    model = create_model(dataset.inputs)
    print(model)

    elite, _ = train_model(
        model,
        dataset,
        destination_directory,
        callback_frequency=10,
    )

    python_writer = GenomeToPython(model.parser)
    python_writer.to_python_class("CoinDetector", elite)

    p, _ = model.predict(dataset.train_x)
    for i, pi in enumerate(p):
        overlayed = draw_overlay(dataset.train_v[i], pi["labels"].astype(np.uint8), color=128, alpha=0.5, thickness=1)  # Instance
        overlayed = draw_overlay(overlayed, pi["markers"].astype(np.uint8), color=255, alpha=0.5, thickness=1)  # Instance
        cv2.imwrite(f"{dataset_path}Instance_segmentation/Results_preview/train{i}final_Cages-task_IS.png", overlayed)


if __name__ == '__main__':
    main()