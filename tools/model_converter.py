"""model_converter
===================
Tools for converting models
Adapted from avelab.common.model_conversion.model_converter

"""

import argparse
import subprocess
import os
import textwrap
import re

from datetime import datetime
from shutil import copy as shutil_copy
from distutils.dir_util import copy_tree

from shutil import rmtree
from enum import Enum
from pathlib import Path
from utils import make_dir, get_md5_hash, write_json

# Versioning schemas, currently we allow the following schemas:
# maj_min_dev: legacy versioning using 3 numbers for major, minor and dev
# a: new versioning schema using 4 numbers for basemodel_arch, basemodel_weighs, submodel_arch, submodel_weights
VersionSchema = Enum("VersionSchema", "maj_min_dev, a")


def convert2ncnn(out_folder, onnx_path, ncnn_path, ncnn_fname="ncnn"):
    """
    Convert a onnx model to ncnn

    :param out_folder: str
        folder where resulting files should be stored
    :param onnx_path: str
        path to onnx simplified and fixed axes
    :param ncnn_path: str
        path to ncnn build dir that contains compiled tools
    :param ncnn_fname: str
        name for the ncnn output files (name.bin, name.param)
    """

    assert os.path.isdir(
        ncnn_path
    ), f"No Ncnn path: {ncnn_path}, is not a dir, can't convert to NCNN"
    print("Converting tensorflow model to ncnn...")
    p = subprocess.Popen(
        [
            os.path.join(ncnn_path, "./bin/onnx2ncnn"),
            onnx_path,
            os.path.join(out_folder, f"{ncnn_fname}.param"),
            os.path.join(out_folder, f"{ncnn_fname}.bin"),
        ],
        cwd=out_folder,
    )
    p.wait()


def optimize_ncnn(out_folder, ncnn_path, in_folder=None, ncnn_fname="ncnn", fp16=False):
    """
    Optimize a ncnn model and convert it to fp16 if requested

    :param out_folder: str
        folder where optimized ncnn files should be saved
    :param ncnn_path: str
        path to ncnn build dir that contains compiled tools
    :param in_folder: str
        path to input folder containing ncnn files (optional, if not specified same as output folder,
        in this case the files will be overwritten)
    :param ncnn_fname: str
        name for the ncnn input and output files (name.bin, name.param)
    :param fp16: bool
        Whether to convert fp16

    """

    assert os.path.isdir(
        ncnn_path
    ), f"No Ncnn path: {ncnn_path}, is not a dir, can't optimize"
    print("Optimizing ncnn model...")
    out_folder = os.path.abspath(out_folder)
    in_folder = os.path.abspath(in_folder if in_folder else out_folder)
    command = [
        os.path.join(ncnn_path, "./bin/ncnnoptimize"),
        os.path.join(in_folder, f"{ncnn_fname}.param"),
        os.path.join(in_folder, f"{ncnn_fname}.bin"),
        os.path.join(out_folder, f"{ncnn_fname}.param"),
        os.path.join(out_folder, f"{ncnn_fname}.bin"),
        "65536" if fp16 else "0",  # 2 ** 16 optimizes model weights to be float16
    ]
    p = subprocess.Popen(
        command,
        cwd=out_folder,
    )
    p.wait()


def convert_onnx(in_folder, out_folder, onnx_fname="model", opset=15):
    """
    Convert a model to onnx

    :param in_folder: str
        path to input folder containing tf files
    :param out_folder: str
        folder where onnx file should be saved
    :param onnx_fname: str
        name for the onnx output file (name.onnx)
    :param opset
        opset used for onnx
    """

    print("Converting tf model to onnx...")
    out_folder = os.path.abspath(out_folder)
    in_folder = os.path.abspath(in_folder)
    p = subprocess.Popen(
        [
            "python3",
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            in_folder,
            "--output",
            os.path.join(out_folder, f"{onnx_fname}.onnx"),
            "--opset",
            f"{opset}",
        ],
    )
    p.wait()


def fix_axes_onnx(
    in_file, input_blob_name, input_shape_str, out_folder, onnx_fname="model"
):
    """
    Fixes axes of an onnx file

    :param in_file: str
        path to input onxx file
    :param input_blob_name: str
        name of the input blob to the model
    :param input_shape_str: str
        input shape as a string comma separated of NCHW
    :param out_folder: str
        folder where onnx file should be saved
    :param onnx_fname: str
        name for the onnx output file (name.onnx)
    """

    print("Fixing onnx model axes...")
    out_folder = os.path.abspath(out_folder)
    in_file = os.path.abspath(in_file)

    p = subprocess.Popen(
        [
            "python3",
            "-m",
            "onnxruntime.tools.make_dynamic_shape_fixed",
            in_file,
            os.path.join(out_folder, f"{onnx_fname}.onnx"),
            "--input_name",
            input_blob_name,
            "--input_shape",
            input_shape_str,
        ],
    )
    p.wait()


def simplify_onnx(in_file, out_folder, onnx_fname="model", num_checks=10):
    """
    Simplifies an onnx file using onnxsim binary. Note that it could also been
    implmented with
    model = onnx.load(in_file)
    model_simp, check = simplify(model)
    but the binary path give further visual output.

    :param in_file: str
        path to input onxx file
    :param out_folder: str
        folder where onnx file should be saved
    :param onnx_fname: str
        name for the onnx output file (name.onnx)
    :param num_checks: int
        Number of checks to verify conversion

    """
    out_folder = os.path.abspath(out_folder)
    in_file = os.path.abspath(in_file)

    print("Simplifying onnx model")
    command = [
        "onnxsim",
        in_file,
        os.path.join(out_folder, f"{onnx_fname}.onnx"),
        f"{num_checks}",
    ]

    p = subprocess.Popen(
        command,
    )
    p.wait()


def process_cpp_header_line(line):
    """Processes a line of the cpp header and return blob name and index (or None, None)

    Parameters
    ----------
    line: str
        line to process

    Returns
    -------
    : str
        blob name
    : int
        blob index

    """
    splitted_line = re.split("\s+", line)
    if (
        splitted_line
        and len(splitted_line) >= 5
        and splitted_line[0] == "const"
        and splitted_line[1] == "int"
        and splitted_line[2][:5] == "BLOB_"
        and splitted_line[3] == "="
        and splitted_line[4][:-1].isnumeric()
    ):
        return splitted_line[2][5:], int(splitted_line[4][:-1])
    else:
        return None, None


def binarize_ncnn(
    out_folder, ncnn_path, in_folder=None, params_fname="ncnn", model_fname="ncnn"
):
    """
    Binarizes a ncnn model

    :param out_folder: str
        folder where optimized ncnn files should be saved
    :param ncnn_path: str
        path to ncnn build dir that contains compiled tools
    :param in_folder: str
        path to input folder containing ncnn files (optional, if not specified same as output folder)
    :param params_fname: str
        name for the ncnn input and output param files (name.param)
    :param model_fname: str
        name for the ncnn input and output model files (name.bin)
    """

    assert os.path.isdir(
        ncnn_path
    ), f"No Ncnn path: {ncnn_path}, is not a dir, can't binarize"
    print("Binarizing ncnn model...")
    out_folder = os.path.abspath(out_folder)
    in_folder = os.path.abspath(in_folder if in_folder else out_folder)
    p = subprocess.Popen(
        [
            os.path.join(ncnn_path, "./bin/ncnn2mem"),
            os.path.join(in_folder, f"{params_fname}.param"),
            os.path.join(in_folder, f"{model_fname}.bin"),
            os.path.join(out_folder, f"{params_fname}.id.h"),
            os.path.join(out_folder, f"{model_fname}.mem.h"),
            os.path.join(out_folder, f"{params_fname}.param.bin"),
        ],
        cwd=out_folder,
    )
    p.wait()


def get_general_model_converter_parser(model_name):
    """
    General model converter parser that contains arguments and help common for all models.

    :param model_name: str
        Name of the model to be converted
    :return: argparse.ArgumentParser
        A general model converter parser object
    """

    parser = argparse.ArgumentParser(
        prog=f"{model_name} model Conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
             Additional Information:

                This has been tested with tensorflow=2.11.0, onnx=1.12.0, tf2onnx=1.14.0/8f8d49, and onnxsim==0.4.13
                Note that other onnxsim version has given problems.

                The script performs the following steps:
                    - Converting a set of pb files and variables into onnx.
                    - Fixing the input dimensions into another onnx file.
                    - Simplifying the onnx file
                    - optimization of ncnn (and potentially conversion to fp16) --> creates ncnn_optimized dir with output files
                    - Binarization of ncnn files
                    - creation of a model bundle
                    - generation of a text file that contains md5 hashes to identify models
                Depending on the settings selected and the paths provided (ave-tracker) not all of these steps
                are performed.

                The output is formatted in the following way:
                     output-models-folder:  --> parent folder (usually in ave-models) in which the model folder will be generated
                        a.2.0.1.0
                            ...
                        a.2.1.1.0:          --> model folder that is automatically generated with the name infered from the version nbr
                            tensorflow_pb:  --> tensorflow models .pb files and variables
                            onnx            --> onnx files
                            ncnn:           --> ncnn files
                            ncnn_optimized  --> ncnn optimized files
                            raw:            --> folder containing all files necessary to create a bundle
                            info.txt        --> text file containing information about the model files
                            bundles         --> contains a bundle intended to be used in ave-tracker
                 NOTE: depending on the settings the structure can be different. If the --keep-intermediate-files is set it will
                        contain all the output files from intermediate steps as well. If some tools are not available,
                        for example the model bundler, the associated  steps will not be performed and the output 
                        folder will be missing the corresponding files.

                Info about additional repositories required:

                ave-tracker:    a compiled version of ave-tracker is required to run the model-bundler
                                and the ncnn binaries that convert the model to ncnn format used in production.
                                Check the our ave-tracker repository on instructions on how to compile it.

                ave-models      a repository containing different versions of our production models. No specific steps
                                required but you might want to check out a new branch for the model that you are
                                converting and then open a PR into master once it's tested. Please also add release 
                                notes containing information about: the training command and procedure, data used, 
                                how this differs from previous versions of the same model as well as a general
                                description of the performance and quality, where the models are stored in the cloud,
                                possibly the  commit hash of the metrabs repository that was used to train the model
                                as well as any other information that will be necessary to the understand and 
                                reproduce the results.
             """
        ),
    )

    parser.add_argument(
        "--output-models-folder",
        type=str,
        required=True,
        help="Path to folder where model folder should be generated,"
        f"normaly {model_name} folder of tracker-models repo",
    )
    parser.add_argument(
        "--version",
        type=int,
        required=True,
        nargs="+",
        help="Version of the model, example: '2 3 1' 3 space separated integers, for legacy models or '4 2 3 1' "
        "4 space separated integers for models using the new versioning schema 'a'",
    )
    parser.add_argument(
        "--version-schema",
        type=str,
        default="a",
        choices=[i.name for i in VersionSchema],
        help="Versioning schema to use.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model that should be converted",
    )
    parser.add_argument(
        "--unoptimized-ncnn",
        action="store_true",
        help="Do not run ncnn optimize",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="When running ncnn optimize use fp16",
    )
    parser.add_argument(
        "--ave-tracker-path",
        type=str,
        default=None,
        help="Path to ave tracker build dir, to run model bundler and ncnn conversion."
        "The usual location is 'ave-tracker/build/client/release'.",
    )
    parser.add_argument(
        "--comment",
        type=str,
        default=None,
        nargs="+",
        help="Additional comment to be added to model notes (optional)",
    )
    parser.add_argument(
        "--keep-intermediate-files",
        action="store_true",
        help="Keeps files from intermediate conversion steps for debugging.",
    )
    parser.add_argument(
        "--string-params",
        action="store_true",
        help="Generates uses strings params file (as opposed to binarized.",
    )
    parser.add_argument(
        "--model-input-name",
        type=str,
        default="input",
        help="Name of input in model",
    )
    parser.add_argument(
        "--input-resolution-hw",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Input resolution as (h, w)",
    )

    return parser


class ModelConverter:
    def __init__(
        self,
        version,
        version_schema,
        out_folder,
        model_dir,
        input_blob_name,
        input_resolution_h_w,
        ave_tracker_path=None,
        optimize=False,
        fp16=False,
        keep_intermediate_files=False,
        binarize_params=False,
        aes_encrypt=False,
    ):
        """
        Initialize a model converter

        Parameters
        ----------
        version: list
            list of integers specifying model version
        version_schema: str
            version schema to use
        out_folder: str
            output folder where the converted model is saved
        model_dir: str
            Path to model that should be converted
        ave_tracker_path: str
            Path to ave-tracker build dir
        optimize: bool
            Runs ncnn optimize
        fp16: bool
            When running ncnn optimize use fp16
        keep_intermediate_files: bool
            keeps all the unused files from intermediate steps if true (for debugging)
        binarize_params: bool
            Binarizes the params files
        aes_encrypt: bool
            encrypt models with aes 128 encryption in the model bundler
        """

        self.out_folder = os.path.abspath(out_folder)
        self.model_dir = os.path.abspath(model_dir)
        self.ave_tracker_path = (
            os.path.abspath(ave_tracker_path) if ave_tracker_path is not None else None
        )
        self.optimize = optimize
        self.fp16 = fp16
        self.keep_intermediate_files = keep_intermediate_files
        self.binarize_params = binarize_params
        self.aes_encrypt = aes_encrypt
        self.input_blob_name = input_blob_name
        self.input_model_height, self.input_model_width = input_resolution_h_w

        self.aux_files = []
        self.settings_json_path = None
        self.ncnn_file_name = "ncnn"
        self.onnx_file_name = "model"

        self.comment = ""
        self.native_model_kind = (
            ""  # model kind used in ave-tracker (only applies to encrypted models)
        )

        self.version_schema = VersionSchema[version_schema]
        self.version_list = version
        self.version_str = ".".join([str(v) for v in version])

        if self.version_schema != VersionSchema.maj_min_dev:
            # add version schema to version string unless we are using the legacy maj_min_dev schema
            self.version_str = f"{version_schema}.{self.version_str}"

        self.conversion_out_folder = os.path.join(self.out_folder, self.version_str)
        self.bundle_folder = os.path.join(self.conversion_out_folder, "bundles")
        self.onnx_folder = os.path.join(self.conversion_out_folder, "onnx")

        self.tf_out_folder = os.path.join(self.conversion_out_folder, "tensorflow_pb")
        self.raw_folder = os.path.join(self.conversion_out_folder, "raw")
        self.ncnn_out_folder = os.path.join(self.conversion_out_folder, "ncnn")
        self.ncnn_optimized_out_folder = os.path.join(
            self.conversion_out_folder, "ncnn_optimized"
        )

        self.clean_up_dirs = (
            []
        )  # list of dirs that get deleted at the end if self.keep_intermediate_files flag is false
        self.final_model_dir = None  # directory to get final models from

        # conversion from pb files without freezing, can be enabled by child classes
        self.conversion_from_pb = False

        # need to be set by child classes
        self.bundle_name = None

    def set_up_folder_structure(self):
        """set up folder structure for model conversion"""
        assert os.path.isdir(self.model_dir), f"invalid model path {self.model_dir}"

        make_dir(self.out_folder)
        make_dir(self.conversion_out_folder, exist_ok=False)
        make_dir(self.tf_out_folder, exist_ok=False)
        make_dir(self.raw_folder, exist_ok=False)
        make_dir(self.onnx_folder, exist_ok=False)

        if self.ave_tracker_path is not None:
            make_dir(self.ncnn_out_folder, exist_ok=False)
            self.clean_up_dirs.append(self.ncnn_out_folder)
            make_dir(self.bundle_folder)

        if self.optimize:
            assert (
                self.ave_tracker_path is not None
            ), "Can not optimize without ncnn path"
            make_dir(self.ncnn_optimized_out_folder, exist_ok=False)
            self.clean_up_dirs.append(self.ncnn_optimized_out_folder)

    def convert_or_load_pb(self):
        """convert a model to generate pb files or load them from model dir"""
        if self.conversion_from_pb:
            self.load_pb_files()
        else:
            self.convert2pb()

    def convert2pb(self):
        """freeze tensorflow model to pb file, needs to be implemented by child class"""
        raise NotImplementedError

    def convert2onnx(self):
        """Converts tensorflow model which is a collection of pb files and variables to onnx"""

        convert_onnx(
            in_folder=self.tf_out_folder,
            out_folder=self.onnx_folder,
            onnx_fname=f"{self.onnx_file_name}_dynamic_axes",
        )

        fix_axes_onnx(
            in_file=os.path.join(
                self.onnx_folder, f"{self.onnx_file_name}_dynamic_axes.onnx"
            ),
            input_blob_name=self.input_blob_name,
            input_shape_str=f"1,3,{self.input_model_height},{self.input_model_width}",
            out_folder=self.onnx_folder,
            onnx_fname=f"{self.onnx_file_name}_fixed_axes",
        )

        simplify_onnx(
            in_file=os.path.join(
                self.onnx_folder, f"{self.onnx_file_name}_fixed_axes.onnx"
            ),
            out_folder=self.onnx_folder,
            onnx_fname=f"{self.onnx_file_name}",
        )

    def convert2ncnn(self):
        """convert tensorflow model to ncnn"""

        convert2ncnn(
            out_folder=self.ncnn_out_folder,
            onnx_path=os.path.join(self.onnx_folder, f"{self.onnx_file_name}.onnx"),
            ncnn_path=self.ave_tracker_path,
            ncnn_fname=self.ncnn_file_name,
        )

        self.final_model_dir = self.ncnn_out_folder

    def optimize_ncnn(self):
        """optimize model with ncnn by converting and potentially convert fp16"""
        assert self.optimize, "Something went wrong, optimization should not be used"

        optimize_ncnn(
            out_folder=self.ncnn_optimized_out_folder,
            ncnn_path=self.ave_tracker_path,
            in_folder=self.ncnn_out_folder,
            ncnn_fname=self.ncnn_file_name,
            fp16=self.fp16,
        )

        self.final_model_dir = self.ncnn_optimized_out_folder

    def binarize_ncnn(self):
        """Binarize model with ncnn creating a .params.bin, id.h and .mem.h files"""
        assert (
            self.binarize_params
        ), "Something went wrong, binarization should not be used"

        binarize_ncnn(
            out_folder=self.final_model_dir,
            ncnn_path=self.ave_tracker_path,
            in_folder=self.final_model_dir,
            params_fname=self.ncnn_file_name,
            model_fname=self.ncnn_file_name,
        )

    def parse_ncnn_header_map(self):
        """Parses the model_name.id.h file
        and return the blob ids as a dictionary"""
        assert (
            self.binarize_params
        ), "Something went wrong, no need to parse header file if not in binarization mode"
        path_header_file = Path(self.final_model_dir) / f"{self.ncnn_file_name}.id.h"
        assert (
            path_header_file.is_file()
        ), f"The header file {path_header_file} not found"
        blobs_map = {}
        with open(path_header_file, "r") as header_file:
            for line in header_file:
                name, index = process_cpp_header_line(line)
                if name is not None:
                    blobs_map[name] = index
        return blobs_map

    def prepare_for_bundler(self):
        """prepare folder for bundler by putting files the correct raw folder"""

        assert self.final_model_dir in [
            self.ncnn_out_folder,
            self.ncnn_optimized_out_folder,
        ]

        for aux_file in self.aux_files:
            shutil_copy(aux_file, self.raw_folder)

        if self.binarize_params:
            # save minimal settings as settings.json in raw folder
            # and copy general settings to tensorflow_pb folder
            write_json(
                os.path.join(self.raw_folder, "settings.json"),
                self.generate_blob_mapping_settings(),
                convert2str=True,
            )
            # attributes model does not have a full settings file
            if self.settings_json_path is not None:
                shutil_copy(self.settings_json_path, self.tf_out_folder)
            shutil_copy(
                os.path.join(self.final_model_dir, f"{self.ncnn_file_name}.param.bin"),
                self.raw_folder,
            )
        else:
            shutil_copy(
                os.path.join(self.final_model_dir, f"{self.ncnn_file_name}.param"),
                self.raw_folder,
            )
            if self.settings_json_path:
                shutil_copy(self.settings_json_path, self.raw_folder)

        shutil_copy(
            os.path.join(self.final_model_dir, f"{self.ncnn_file_name}.bin"),
            self.raw_folder,
        )

    def make_model_bundle(self):
        """make model bundle using ave-tracker model bundler"""

        assert (
            self.ave_tracker_path is not None
        ), "Can't make model bundle without ave-tracker and model bundler"
        assert self.bundle_name is not None, "No bundle name set!"

        argumment_list = [
            os.path.join(self.ave_tracker_path, "bin/model_bundler"),
            "-i",
            self.raw_folder,
            "-o",
            f"./{self.bundle_name}",
        ]

        if self.aes_encrypt:
            assert (
                self.native_model_kind
            ), "native model kind needs to be defined to encrypt models"
            argumment_list.append("--encrypt")
            argumment_list.extend(["--model-kind", self.native_model_kind])

        if self.version_schema == VersionSchema.maj_min_dev:
            assert (
                len(self.version_list) == 3
            ), f"versioning schema 'maj_min_dev' requires 3 entries but got: {self.version_list}"
            argumment_list.extend(["--version-id", self.version_str])
        elif self.version_schema == VersionSchema.a:
            assert (
                len(self.version_list) == 4
            ), f"versioning schema 'a' requires 4 entries but got: {self.version_list}"
            argumment_list.extend(["--version-id", self.version_str])
        else:
            raise ValueError(f"Invalid version schema: {self.version_schema}")

        p = subprocess.Popen(argumment_list, cwd=self.bundle_folder)
        p.wait()

    def generate_info_file(self):
        """generate a text file with information about the model
        - time of conversion
        - if model was optimized with ncnn
        - md5 hash of original .pb models
        - md5 hash of converted .bin and .param files
        - optional comment

        """

        lines = list()
        lines.append("time of conversion: {}".format(str(datetime.now())))
        lines.append("ncnn optimization: {}".format(str(self.optimize)))

        tf_folder = Path(self.tf_out_folder)
        tf_files = [
            f for f in tf_folder.glob("**/*") if f.is_file() and f.name != ".DS_Store"
        ]
        for file_ in tf_files:
            lines.append(
                f"md5 hash of {str(file_.relative_to(tf_folder))} file: {get_md5_hash(str(file_))}"
            )

        onnx_file_path = os.path.join(self.onnx_folder, f"{self.onnx_file_name}.onnx")
        lines.append(
            f"md5 hash of onnx file {self.onnx_file_name}.onnx: {get_md5_hash(onnx_file_path)}"
        )

        if self.final_model_dir:
            bin_file = os.path.join(self.raw_folder, f"{self.ncnn_file_name}.bin")
            lines.append(f"md5 hash of bin file: {get_md5_hash(bin_file)}")
            param_suffix = ".param.bin" if self.binarize_params else ".param"
            param_file = os.path.join(
                self.raw_folder, f"{self.ncnn_file_name}{param_suffix}"
            )
            lines.append(f"md5 hash of param.bin file: {get_md5_hash(param_file)}")
        if self.comment is not None:
            lines.append("comment: {}".format(self.comment))

        with open(os.path.join(self.conversion_out_folder, "info.txt"), "w") as f:
            for line in lines:
                f.write(line + "\n")

    def convert(self):
        """method to perform all the conversion steps"""

        # set up folder structure
        self.set_up_folder_structure()

        # convert to pb or use existing pb files
        self.convert_or_load_pb()

        # convert to pb files to onnx
        self.convert2onnx()

        # convert onnx to ncnn
        if self.ave_tracker_path is not None:
            self.convert2ncnn()

            # optimize
            if self.optimize:
                self.optimize_ncnn()

            if self.binarize_params:
                self.binarize_ncnn()

            # prepare directory for bundling
            if self.final_model_dir:
                self.prepare_for_bundler()

            # add release notes
            self.generate_info_file()

            # make model bundle
            self.make_model_bundle()

    def __del__(self):
        """clean up unnecessary files if desired"""
        if not self.keep_intermediate_files and self.clean_up_dirs:
            print("cleaning up directory")
            for dir_path in self.clean_up_dirs:
                print(f"removing: {dir_path}")
                rmtree(dir_path, ignore_errors=False, onerror=None)

    def generate_blob_mapping_settings(self):
        """generates minimal settings dict with blob mappings, needs to be implemented by child class if binary params should be supported"""
        raise NotImplementedError

    def load_pb_files(self):
        """loading pb files for model conversion from preexisting pb files without freezing"""
        source_model_dir = Path(self.model_dir)

        # check is a model modlder
        source_pb_files = [
            f
            for f in source_model_dir.glob("**/*")
            if f.is_file() and f.name.endswith(".pb")
        ]
        assert (
            len(source_pb_files) == 3
        ), f"Expected to find three files in {source_model_dir} but found {len(source_pb_files)}"

        # TODO: add check that variables folder exists

        copy_tree(source_model_dir, self.tf_out_folder)
