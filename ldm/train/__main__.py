import ldm
import torch
import wandb
import pytorch_lightning as pl
import os
from pytorch_lightning.loggers import WandbLogger

torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    # PARSER
    parser = ldm.get_training_parser()
    args = parser.parse_args()

    # SEED
    pl.seed_everything(args.seed, workers=True)

    # DATAMODULE
    data_config = ldm.get_data_config(args=args)
    datamodule = ldm.get_datamodule(
        config=data_config,
        dataset=args.dataset
    )

    # WANDB (OPTIONAL)
    if args.wandb is not None:
        wandb.login(key=args.wandb)  # API KEY
        name = args.name or f"ldm-{args.stage}"
        logger = WandbLogger(project="latent-diffusion-model", name=name, log_model=False)
    else:
        logger = None

    # MODEL
    if args.model_ckpt is None:
        model_config = ldm.get_model_config(args)
        model_config.in_channels = datamodule.in_channels
        model = ldm.LatentDiffusionModel(model_config)
        if args.stage == "stage2":
            model.load_vae_ckpt(args.vae_ckpt)
    else:
        model = ldm.LatentDiffusionModel.load_from_checkpoint(args.model_ckpt)
        print("Model Checkpoint Loaded!")
    # CALLBACK
    root_path = os.path.join(os.getcwd(), "checkpoints")
    callback = ldm.ModelCallback(
        root_path=root_path,
        stage=args.stage
    )
    # STRATEGY
    strategy = 'ddp_find_unused_parameters_true' if torch.cuda.is_available() else 'auto'

    # TRAINER
    trainer = pl.Trainer(
        default_root_dir=root_path,
        logger=logger,
        callbacks=callback.get_callback(),
        strategy=strategy,
        gradient_clip_val=1.0,
        max_epochs=data_config.max_epochs,
        enable_progress_bar=args.pbar,
        deterministic=False,
        precision=args.precision,
        accumulate_grad_batches=max(int(data_config.max_batch_size / data_config.batch_size), 1)
    )

    # FIT MODEL
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()
