import ldm


def get_datamodule(dataset: str = "celebahq", **kwargs):
    if dataset == "celebahq":
        return ldm.CelebADataModule(**kwargs)
    elif dataset == "mnist":
        return ldm.MNISTDataModule(**kwargs)
    else:
        raise ValueError("dataset must be in ['celebahq','mnist']")
