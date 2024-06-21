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
from kartezio.endpoint import EndpointWatershed

dataset_path = "/Users/mathieu/Desktop/Kartezio/Cages-task/"

def create_model(n_inputs):
    n_iterations = 20_000
    n_children = 5
    n_nodes = 30
    model = create_segmentation_model(
        n_iterations,
        n_children,
        inputs=n_inputs,
        nodes=n_nodes,
        #fitness=FitnessAP(0.5)
    )
    return model

def main():
    destination_directory = './models'
    dataset = read_dataset(dataset_path, filename="dataset_cages-task.csv", meta_filename="META_cages-task.JSON", preview=True)
    train_x, train_y, train_v = dataset.train_xyv
    # test_x, test_y, test_v = dataset.test_xyv  ## Added
    # print(test_x, test_y, test_v)
    # for i in range(len(train_x)):
    #     print(train_v[i].shape, train_y[i][0].shape)  ## Added during TP
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
    # p, _ = model.predict(dataset.test_x)
    for i, pi in enumerate(p):
        overlayed = draw_overlay(dataset.train_v[i], pi["mask"].astype(np.uint8), color=[128, 0, 0], alpha=0.6, thickness=2)  # Segmentation
        # overlayed = draw_overlay(dataset.test_v[i], pi["mask"].astype(np.uint8), color=[128, 0, 0], alpha=0.6, thickness=2)  # Segmentation
        cv2.imwrite(f"{dataset_path}Semantic_segmentation/Results_preview/test{i}final_Cages-task_SS.png", overlayed)


if __name__ == '__main__':
    main()