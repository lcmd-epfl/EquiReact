import pandas as pd
import numpy as np
import torch
from lightning import pytorch as pl
from pathlib import Path

from chemprop import data, featurizers, models, nn
import argparse

def train_chemprop_rxn(csv_name, smiles_column,target_columns, featurizer="PROD_DIFF"):
    chemprop_dir = Path.cwd().parent
    df = pd.read_csv(csv_name)
    input_path = chemprop_dir / "tests" / "data" / "regression" / "rxn" / "rxn.csv"
    num_workers = 0  # number of workers for dataloader. 0 means using main process for data loading
    smis = df.loc[:, smiles_column].values
    ys = df.loc[:, target_columns].values

    all_data = [data.ReactionDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

    # Split data
    mols = [d.rct for d in all_data]  # Can either split by reactants (.rct) or products (.pdt)
    train_indices, val_indices, test_indices = data.make_split_indices(mols, "random", (0.8, 0.1, 0.1))
    train_data, val_data, test_data = data.split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )

    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_=featurizer)

    train_dset = data.ReactionDataset(train_data[0], featurizer)
    scaler = train_dset.normalize_targets()

    val_dset = data.ReactionDataset(val_data[0], featurizer)
    val_dset.normalize_targets(scaler)
    test_dset = data.ReactionDataset(test_data[0], featurizer)

    train_loader = data.build_dataloader(train_dset, num_workers=num_workers)
    val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
    test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)



    fdims = featurizer.shape # the dimensions of the featurizer, given as (atom_dims, bond_dims).
    mp = nn.BondMessagePassing(*fdims)

    print(nn.agg.AggregationRegistry)

    agg = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform)
    batch_norm = True

    metric_list = [nn.metrics.RMSE(), nn.metrics.MAE()]

    # construct MPNN

    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    # set up trainer
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,  # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=20,  # number of epochs to train for
    )

    # train model
    trainer.fit(mpnn, train_loader, val_loader)

    # test model
    results = trainer.test(mpnn, test_loader)
    print(results)


def predict_chemprop_rxn(model_path, csv_name, smiles_column, featurizer="PROD_DIFF"):


    chemprop_dir = Path.cwd().parent
    checkpoint_path = chemprop_dir / "tests" / "data" / "example_model_v2_regression_rxn.ckpt" # path to the checkpoint file.
    # If the checkpoint file is generated using the training notebook, it will be in the `checkpoints` folder with name similar to `checkpoints/epoch=19-step=180.ckpt`.
    mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)
    df = pd.read_csv(csv_name)
    smis = df.loc[:, smiles_column].values
    all_data = [data.ReactionDatapoint.from_smi(smi) for smi in smis]
    featurizer = featurizers.CondensedGraphOfReactionFeaturizer(mode_=featurizer)
    dset = data.ReactionDataset(all_data, featurizer)
    loader = data.build_dataloader(dset, shuffle=False)

    # perform tests
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="cpu",
            devices=1
        )
        test_preds = trainer.predict(mpnn, loader)

    preds = np.concatenate(test_preds, axis=0)
    return preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='train', help='train / predict')
    parser.add_argument('-d', '--data', default='data/juliette/CMD_TS_smiles_ok.csv', help='data csv file')
    parser.add_argument('-s', '--smiles_column', default='Int1_to_Int2_SMILES', help='smiles column name')
    parser.add_argument('-t', '--target_columns', nargs='+', default=['fw_td_kcalmol'], help='target column names (for training)')
    parser.add_argument('-f', '--featurizer', default='PROD_DIFF', help='featurizer: PROD_DIFF / PROD / REACT_DIFF / REACT')
    parser.add_argument('-mp', '--model_path', default='model.ckpt', help='model checkpoint path (for prediction)')
    args = parser.parse_args()

    if args.mode == 'train':
        train_chemprop_rxn(args.data, args.smiles_column, args.target_columns, featurizer=args.featurizer)
    elif args.mode == 'predict':
        preds = predict_chemprop_rxn(args.model_path, args.data, args.smiles_column, featurizer=args.featurizer)
        print(preds)
    else:
        raise ValueError(f'Unknown mode {args.mode}')