from pathlib import Path
import json
from speechbrain.pretrained import SpeakerRecognition
import argparse


def mine_voice(sample_voice: Path, lookup_path: Path, output_path: Path):
    output_path.mkdir(exist_ok=True, parents=True)
    save_path = output_path / "save.json"

    if not save_path.exists():
        save_path.write_text("{}")

    saves = json.load(save_path.open("r", encoding="utf-8"))

    key = str(sample_voice)
    if key not in saves:
        saves[key] = {}

    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
    )

    for speech in lookup_path.iterdir():
        choose = False
        speech_key = str(speech)
        if speech_key in saves[key]:
            choose = saves[key][speech_key]
        else:
            score, prediction = verification.verify_files(
                str(sample_voice), str(speech)
            )
            choose = 0.7 <= score.item()  # and score.item() <= 0.7
            if choose:
                speech.rename(output_path / speech.name)
            saves[key][speech_key] = choose
            json.dump(saves, save_path.open("w", encoding="utf-8"))
        print(
            f"Verified {speech} | Score {score.item()} | Prediction {prediction.item()} | Choose {choose}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Find similar voices from a voice sample."
    )
    parser.add_argument(
        "-s", "--sample", help="Path to the reference voice", type=str, required=True
    )
    parser.add_argument(
        "-l", "--lookup", help="Path to the folder to look up", type=str, required=True
    )
    parser.add_argument(
        "-o", "--output", help="Path to the output folder", type=str, required=True
    )

    args = parser.parse_args()

    sample_voice = Path(args.sample)
    lookup_path = Path(args.lookup)
    output_path = Path(args.output)
    # not checking whether exists or is a file or folder blah blah blah ...

    mine_voice(
        sample_voice=sample_voice, lookup_path=lookup_path, output_path=output_path
    )


if __name__ == "__main__":
    main()
