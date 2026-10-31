import argparse
import json
from fraud_engine.data import ensure_data, generate_data, RAW
from fraud_engine.train import train
from fraud_engine.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Fraud & Anomaly Detection Engine")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate-data")
    gen.add_argument("--rows", type=int, default=12000)
    commands.add_parser("train")
    commands.add_parser("evaluate")
    args = parser.parse_args()
    if args.command == "generate-data":
        RAW.parent.mkdir(parents=True, exist_ok=True)
        generate_data(args.rows).to_csv(RAW, index=False)
        print(f"Generated {args.rows} rows at {RAW}")
    elif args.command == "train":
        print(json.dumps(train(), indent=2))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(), indent=2))


if __name__ == "__main__":
    main()
