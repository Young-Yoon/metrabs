import os.path
import numpy as np
import cameralib
import data.datasets3d as ps3d
import paths
import util
from options import logger
from data.utils import visualize_sample


def get_sequence_info(
    phase, root_dataset, ratio_phase_train=0.8, ratio_phase_validation=0.1
):
    """Returns a list of sequences in the dataset for a given phase/split.
    It uses 80% of the dataset as train, 10% for validation, and 10% for testing.

    Args:
        phase: str
            Corresponding split to find sequences
        root_dataset: str
            root path of the dataset
        ratio_phase_train: float
            Ratio of sequences for the train phase/split
        ratio_phase_validation: float
            Ratio of sequences for the validation phase/split
    Returns:
        List of relative paths to sequences

    """
    existing_subjects = [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
    ]
    if phase == "train":
        subjects = existing_subjects[: int(len(existing_subjects) * ratio_phase_train)]
    elif phase == "validation":
        subjects = existing_subjects[
            int(len(existing_subjects) * ratio_phase_train) : int(
                len(existing_subjects) * (ratio_phase_train + ratio_phase_validation)
            )
        ]
    elif phase == "test":
        subjects = existing_subjects[
            int(len(existing_subjects) * (ratio_phase_train + ratio_phase_validation)) :
        ]
    else:
        raise Exception(
            f"phase {phase} is not supported. Restrict yourself to train, validation, test."
        )

    seq_folders = list()
    for subject in subjects:
        seq_folders += [
            os.path.join(subject, sequence)
            for sequence in os.listdir(os.path.join(root_dataset, subject))
            if os.path.isdir(os.path.join(root_dataset, subject, sequence))
        ]

    return seq_folders


def load_sequence_annotations(sequence, root_dataset):
    """For a given sequence returns the annotations

    Args:
        sequence: str
            relative path for a sequence in the dataset
        root_dataset: str
            root path of the dataset

    Returns: tuple(bool, None/ tuple(camera, world_pose3d, bbox)
        Whether there is annotation or not for the dataset
        Annotations

    """
    sequence_full_path = os.path.join(root_dataset, sequence)
    fi, fe, fw, fb = (
        os.path.join(sequence_full_path, "metrab_annotations", f)
        for f in ["intrinsics.npy", "extrinsics.npy", "poses3d.npy", "bbox.npy"]
    )
    if not os.path.exists(fi):
        return False, None

    intrinsics, extrinsics = np.load(fi, allow_pickle=True), np.load(
        fe, allow_pickle=True
    )

    assert len(intrinsics.shape) == 2
    camera = cameralib.Camera(
        extrinsic_matrix=extrinsics, intrinsic_matrix=intrinsics, world_up=(0, 1, 0)
    )

    world_pose3d = np.load(fw, allow_pickle=True).item()
    bbox = np.load(fb, allow_pickle=True).item()

    return True, (camera, world_pose3d, bbox)


def get_examples(phase, visualize=False):
    """Gets the different samples of the dataset for the corresponding phase or split.

    Args:
        phase: str
            Corresponding split to load
        visualize: bool
            Whether to use dump example to disk

    Returns: list(Pose3DExample)
        List of Pose3DExample examples

    """
    result = []
    # From 'pelv,rhip,rkne,rank,lhip,lkne,lank,spin,neck,head,htop,lsho,lelb,lwri,rsho,relb,rwri’ --> root-last
    i_relevant_joints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0]
    root_dataset = f"{paths.DATA_ROOT}/pandora"

    for sequence in util.progressbar(get_sequence_info(phase, root_dataset)):
        load_success, params = load_sequence_annotations(sequence, root_dataset)
        if not load_success:
            continue
        camera, world_pose3d, bbox = params

        for frame in world_pose3d.keys():
            if frame not in world_pose3d.keys():
                continue  # this should not happen

            world_coords = world_pose3d[frame][i_relevant_joints, :]
            bbox_fr = bbox[frame]
            impath = os.path.join("pandora", sequence, "RGB", f"{frame}.png")
            ex = ps3d.Pose3DExample(impath, world_coords, bbox=bbox_fr, camera=camera)

            if visualize:
                visualize_sample(
                    os.path.join(paths.DATA_ROOT, impath),
                    bbox=bbox_fr,
                    out_path=f"{frame}.jpg",
                )

            result.append(ex)
    return result


@util.cache_result_on_disk(
    f"{paths.CACHE_DIR}/pandora.pkl", min_time="2023-06-27T11:30:43"
)
def make_pandora():
    """Reads the local pandora dataset or fetches it from cache as pkl file.

    Returns: ps3d.Pose3DDataset
        Returns a Pose3DDataset representing pandora.

    """
    joint_names = (
        "rhip,rkne,rank,lhip,lkne,lank,tors,neck,head,htop,"
        "lsho,lelb,lwri,rsho,relb,rwri,pelv".split(",")
    )

    edges = (
        "htop-head-neck-lsho-lelb-lwri,neck-rsho-relb-rwri,"
        "neck-tors-pelv-lhip-lkne-lank,pelv-rhip-rkne-rank"
    )

    joint_info = ps3d.JointInfo(joint_names, edges)

    train_examples = get_examples("train")
    valid_examples = get_examples("validation")
    test_examples = get_examples("test")

    logger.info(f"Pandora number train samples: {len(train_examples)}")
    logger.info(f"Pandora number validation samples: {len(valid_examples)}")
    logger.info(f"Pandora number test samples: {len(test_examples)}")

    train_examples.sort(key=lambda x: x.image_path)
    valid_examples.sort(key=lambda x: x.image_path)
    test_examples.sort(key=lambda x: x.image_path)
    return ps3d.Pose3DDataset(joint_info, train_examples, valid_examples, test_examples)
