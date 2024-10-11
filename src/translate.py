import pandas as pd

from pathlib import Path
from mtranslate import translate
from tqdm.contrib.concurrent import thread_map
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data_dir", type=str, default="public_data/train/track_a")
parser.add_argument("--output_dir", type=str, default="synthetic_data/train/track_a")
parser.add_argument("--language", type=str, required=True)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()


def main():
    path = Path(args.data_dir)
    output_path = Path(args.output_dir)
    csv_file = path / f"{args.language}.csv"
    output_csv = output_path / f"{args.language}_tn.csv"
    df = pd.read_csv(csv_file)

    language = csv_file.stem
    language2code = {
        "afr": "af",
        "amh": "am",
        "deu": "de",
        "eng": "en",
        "oro": "om",
        "ptbr": "pt",
        "rus": "ru",
        "som": "so",
        "tir": "ti",
    }

    translate_fn = lambda text: translate(text, to_language="su", from_language=language2code[language])

    translated_texts = thread_map(translate_fn, df["text"], max_workers=args.num_workers)
    df["text_tn"] = translated_texts
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
