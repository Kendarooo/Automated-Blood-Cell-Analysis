from utils.config import load_config
from utils.seed import set_seed


def main() -> None:
    config = load_config()
    set_seed(config["seed"])

    print("Proyecto base funcionando")
    print("Seed:", config["seed"])
    print("Clases:", config["dataset"]["class_names"])
    print("Extractor:", config["feature_extractor"]["backbone"])
    print("Corte:", config["feature_extractor"]["truncate_at"])


if __name__ == "__main__":
    main()