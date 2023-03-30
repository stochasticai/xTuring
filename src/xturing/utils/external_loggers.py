import os
import warnings

import transformers


def configure_external_loggers():
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    transformers.logging.set_verbosity_error()
    warnings.filterwarnings("ignore")
