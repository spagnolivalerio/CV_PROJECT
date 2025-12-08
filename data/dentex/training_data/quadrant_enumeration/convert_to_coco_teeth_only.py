import json
import argparse
from pathlib import Path

def convert_json(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r") as f:
        data = json.load(f)

    # Controllo minimo struttura
    assert "images" in data, "Manca la chiave 'images' nel JSON"
    assert "annotations" in data, "Manca la chiave 'annotations' nel JSON"
    assert "categories_2" in data, "Manca la chiave 'categories_2' nel JSON"

    images = data["images"]
    annotations = data["annotations"]
    categories_2 = data["categories_2"]

    new_annotations = []
    for ann in annotations:
        ann_new = ann.copy()

        # Rimuovi la categoria 1 se presente
        if "category_id_1" in ann_new:
            ann_new.pop("category_id_1")

        # Prendi category_id_2 e rinominalo in category_id
        if "category_id_2" not in ann_new:
            raise ValueError(f"Annotazione senza 'category_id_2': {ann_new}")

        ann_new["category_id"] = ann_new.pop("category_id_2")

        new_annotations.append(ann_new)

    # Costruisci nuovo dizionario COCO-valido
    new_data = {
        "images": images,
        "annotations": new_annotations,
        "categories": categories_2,  # rinomina categories_2 → categories
    }

    with output_path.open("w") as f:
        json.dump(new_data, f, indent=2)

    print(f"Conversione completata.")
    print(f"  Immagini:     {len(images)}")
    print(f"  Annotazioni:  {len(new_annotations)}")
    print(f"  Categorie:    {len(categories_2)}")
    print(f"File salvato in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte il JSON del dataset tenendo solo il tipo di dente (categories_2) in formato COCO valido."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="train_quadrant_enumeration.json",
        help="Path del JSON originale",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="train_teeth_only_coco.json",
        help="Path del JSON di output",
    )

    args = parser.parse_args()
    convert_json(args.input, args.output)
