import yaml

from transformer import StepTransformer

if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    features = params["data"]["features"]
    target_column = params["data"]["target_column"]
    raw_path = params["data"]["raw_path"]

    transformer = StepTransformer(columns=features + [target_column], target_column=target_column)
    transformer.transform(raw_path)
