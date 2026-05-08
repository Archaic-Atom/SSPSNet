# -*coding: utf-8 -*-
import os
import JackFramework as jf
from UserModelImplementation.user_interface import UserInterface
import warnings


def main() -> None:
    app = jf.Application(UserInterface(), "your network name")
    app.start()


# execute the main function
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    main()
