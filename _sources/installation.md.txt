# Installation

Detex only runs on old versions of python and associated libraries, and is no longer being actively maintained. However, I have tried to make it as easy as possible to still want to use it. I will also certainly be willing to review pull requests if you want to make any changes to the code.

In order to install detex you **must** create a new conda environment using the requirements file found [here](https://github.com/d-chambers/Detex/blob/master/detex_env.txt). You can either download it directly, clone the repo, or copy and past it into a new file.

Next, create a new environment named detex and use the detex_env.txt file like so:

```bash
conda create -n detex --file detex_env.txt
```

Now you are ready to run the tutorial notebooks.