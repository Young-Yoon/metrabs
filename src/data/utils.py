def visualize_sample(
    imagepath, projected_2d=None, bbox=None, keypoint_2d=None, out_path="sample.jpg"
):
    """Visualize a dataset sample and dumps it to disk

    Args:
        imagepath: str
            Path to the image
        projected_2d:
            2D projection of 3D joints
        bbox:
            Bounding box
        keypoint_2d:
            2d keypoints
        out_path: str
            output path where to dump the file

    """
    import matplotlib.pyplot as plt
    from skimage.io import imread, imshow

    image = imread(imagepath)
    h, w, _ = image.shape
    plt.imshow(image)

    ## Show the reprojected keypoints
    if projected_2d is not None:
        for j in range(projected_2d.shape[0]):
            plt.plot(
                projected_2d[j, 0],
                projected_2d[j, 1],
                "o",
                markersize=7,
                color="orange",
            )

    ### Show the keypoints
    if keypoint_2d is not None:
        for joint2d in keypoint_2d:
            x = joint2d["u"] * w
            y = joint2d["v"] * h
            plt.plot(
                x, y, "o", markersize=3, color="white", alpha=joint2d["confidence"]
            )
    if bbox is not None:
        min_x = bbox[0]
        min_y = bbox[1]
        max_x = bbox[0] + bbox[2]
        max_y = bbox[1] + bbox[3]
        plt.plot(
            [min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y]
        )
    plt.show()
    plt.savefig(out_path)
