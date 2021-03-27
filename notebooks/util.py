import matplotlib.pyplot as plt
import pandas as pd

from qiime2 import Artifact, Metadata
from qiime2.plugins import songbird

table = Artifact.load("../data/oral_trimmed_deblur.qza")
metadata = pd.read_csv("../data/oral_trimmed_metadata.txt", sep="\t",
                       index_col=0)
metadata.columns = [x.replace(" ", "_") for x in metadata.columns]
metadata.columns = [x.replace(")", "") for x in metadata.columns]
metadata.columns = [x.replace("(", "") for x in metadata.columns]
metadata = Metadata(metadata)

def run_songbird(dp, lr, epochs, formula="C(brushing_event)"):
    diff, stats, biplot = songbird.actions.multinomial(
        table,
        metadata,
        formula,
        training_column="Test",
        differential_prior=dp,
        learning_rate=lr,
        epochs=epochs,
        summary_interval=0.5,
        batch_size=3,
        random_seed=42,
        silent=True
    )
    return diff, stats, biplot


def get_stats_df(stats):
    fp = str(stats._archiver.data_dir) + "/stats.tsv"
    df = pd.read_csv(fp, sep="\t", index_col=0, skiprows=[1])
    return df


def plot_graphs(stats_df):
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axs[0].plot(stats_df["iteration"], stats_df["cross-validation"])
    axs[1].plot(stats_df["iteration"], stats_df["loss"])
    axs[1].set_xlabel("Iteration", fontsize=16)
    axs[0].set_ylabel("Cross-Validation", fontsize=16)
    axs[1].set_ylabel("Loss", fontsize=16)
    return axs
