"""metrabs_converter
====================

Converts a metrabs model for use in ave-tracker.

example command:
python tools/metrabs_converter.py \
  --output-models-folder ~/git/ave-models/body/ \
   --version 2 1 1 4 \
   --model-path ~/Downloads/160in_ms_w1_up_pv05_abs0_scratch/model \
   --model-kind Metrabs \
   --ave-tracker-path /Users/inavarro/git/ave-tracker/build/client/release \
   --keep-intermediate-files \
   --input-resolution-hw 160 160 \
   --model-input-name input_2


This has been tested with tensorflow=2.11.0, onnx=1.12.0, tf2onnx=1.14.0/8f8d49, and onnxsim==0.4.13
Note that other onnxsim version has given problems.

"""

from pathlib import Path
from enum import Enum
from model_converter import (
    ModelConverter,
    get_general_model_converter_parser,
)


# Metrabs model kind enum to identify which kind of model is being converted
MetrabsModelKind = Enum("MetrabsModelKind", "Metrabs")


def parse_arguments():
    """
    Parse command line arguments and provide help.
    """

    parser = get_general_model_converter_parser("Metrabs")
    parser.add_argument(
        "--model-kind",
        type=str,
        required=True,
        choices=[i.name for i in MetrabsModelKind],
        help="Kind of Metrabs model to be converted.",
    )
    args = parser.parse_args()

    return args


class MetrabsModelConverter(ModelConverter):
    def __init__(
        self,
        version,
        version_schema,
        out_models_folder,
        model_dir,
        input_blob_name,
        input_resolution_h_w,
        ave_tracker_path,
        optimize,
        fp16,
        model_kind,
        binarize_params,
        comment=None,
        keep_intermediate_files=False,
    ):
        super().__init__(
            version,
            version_schema,
            out_models_folder,
            model_dir,
            input_blob_name,
            input_resolution_h_w,
            ave_tracker_path,
            optimize,
            fp16,
            keep_intermediate_files,
            binarize_params=binarize_params,
            aes_encrypt=True,  # all metrabs models should be aes encrypted
        )
        self.model_kind = model_kind

        self.ncnn_file_name = "metrabs"
        self.onnx_file_name = "metrabs"
        self.bundle_name = "bodypose_bundle.deploy"
        self.native_model_kind = "Bodypose"

        self.conversion_from_pb = True
        self.raw_folder = str(Path(self.conversion_out_folder) / "raw")

        self.comment = " ".join(comment) if comment is not None else None

    def generate_blob_mapping_settings(self):
        """generates minimal settings dict with blob mappings which is used for models with binarized param"""

        blobs_map = self.parse_ncnn_header_map()

        # some of the settings hardcoded now should be read from onnx
        effective_stride = 32
        settings = {
            "kInputSize": f"{self.input_model_width}",
            "kOutHW": f"{round(self.input_model_width/effective_stride)}",
            "kOutC": "72",
            "kNKeypoints": "8",
            "kNJoints": "8",
            "kInputLayer": f"{blobs_map[self.input_blob_name]}",
            "kOutputLayer": f"{blobs_map['output_1']}",
        }
        return settings


def main():
    # parse cmd line arguments
    args = parse_arguments()

    # create ModelConverter object
    model_converter = MetrabsModelConverter(
        version=args.version,
        version_schema=args.version_schema,
        out_models_folder=args.output_models_folder,
        model_dir=args.model_path,
        input_blob_name=args.model_input_name,
        input_resolution_h_w=args.input_resolution_hw,
        ave_tracker_path=args.ave_tracker_path,
        optimize=not args.unoptimized_ncnn,
        fp16=args.fp16,
        model_kind=MetrabsModelKind[args.model_kind],
        binarize_params=not args.string_params,
        comment=args.comment,
        keep_intermediate_files=args.keep_intermediate_files,
    )

    # convert Model
    model_converter.convert()


if __name__ == "__main__":
    main()
