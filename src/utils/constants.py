SETUP_FILE = "src/arch/setup.yml"
SPLIT_SET = "data/CrisisMMD_v2.0_baseline_split/event_data_splits/{}/splits/{}_s{}_{}.jsonl"
AL_SPLIT_SET = "data/CrisisMMD_v2.0_baseline_split/event_data_splits/{}/al_splits/{}/{}_s{}_{}.jsonl"
DEV_SET = "data/CrisisMMD_v2.0_baseline_split/event_data_splits/{}/dev.jsonl"
TRAIN_SET = "data/CrisisMMD_v2.0_baseline_split/event_data_splits/{}/train.jsonl"
RESULT_FILE = "results/{}/{}/{}/{}_{}_{}.json"
WANDB_NAME = "exp_{}_param_{}_{}_{}_{}.json"
IMAGEPATH = "./data/CrisisMMD_v2.0/"
DATAPATH = "./data/CrisisMMD_v2.0_baseline_split/data_splits/informative_orig/"
EVENTS = ["california_wildfires", "hurricane_harvey", "hurricane_irma", "hurricane_maria", "iraq_iran_earthquake", "mexico_earthquake", "srilanka_floods"]
RED_CACHE = "data/.cache/{}_{}_{}.pkl"
GRAPH_CACHE = "data/.cache/graph_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl"
AUG_GRAPH_CACHE = "data/.cache/aug_graph_{}_aug_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl"
EVENT_SPLIT_JSON = "data/CrisisMMD_v2.0_baseline_split/event_data_splits/{}/{}.jsonl"
EVENT_AUG_PAIRS = {
    "california_wildfires": "hurricane_harvey",
    "hurricane_harvey": "iraq_iran_earthquake",
    "iraq_iran_earthquake": "hurricane_irma",
    "hurricane_irma": "mexico_earthquake",
    "mexico_earthquake": "hurricane_maria",
    "hurricane_maria": "srilanka_floods",
    "srilanka_floods": "california_wildfires"
}