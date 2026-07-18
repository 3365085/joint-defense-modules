from pathlib import Path

from model_security_gate.utils.io import label_path_for_image, read_yolo_labels


def test_nested_yolo_labels_preserve_image_subdirectories(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "val" / "A" / "sample.png"
    labels_dir = tmp_path / "labels" / "val"
    label_path = labels_dir / "A" / "sample.txt"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    label_path.write_text("1 0.5 0.5 0.2 0.4\n", encoding="utf-8")

    assert label_path_for_image(image_path, labels_dir) == label_path
    labels = read_yolo_labels(image_path, (100, 200, 3), labels_dir=labels_dir)

    assert len(labels) == 1
    assert labels[0]["cls_id"] == 1
