import ldm


def get_datamodule(config: ldm.BaseCelebAHQConfig, dataset: str = "celebahq"):
    if dataset == "celebahq":
        return ldm.CelebADataModule(config)
    elif dataset == "mnist":
        return ldm.MNISTDataModule(config)
    else:
        raise ValueError("dataset must be in ['celebahq','mnist']")
