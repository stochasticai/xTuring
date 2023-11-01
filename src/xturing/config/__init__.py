import torch

from xturing.utils.interactive import is_interactive_execution
from xturing.utils.logging import configure_logger
from xturing.utils.utils import assert_install_itrex

logger = configure_logger(__name__)

# check if cuda is available, if not use cpu and throw warning
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DTYPE = torch.float16 if DEFAULT_DEVICE.type == "cuda" else torch.float32
IS_INTERACTIVE = is_interactive_execution()

if DEFAULT_DEVICE.type == "cpu":
    logger.warning("WARNING: CUDA is not available, using CPU instead, can be very slow")


def assert_not_cpu_int8():
    assert DEFAULT_DEVICE.type != "cpu", "Int8 models are not supported on CPU"

def assert_cpu_int8_on_itrex():
    if DEFAULT_DEVICE.type == "cpu":
        assert_install_itrex()