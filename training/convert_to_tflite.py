import argparse
import tensorflow as tf

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SavedModel to TFLite")
    parser.add_argument("--saved-model", default="scoreboard_saved_model")
    parser.add_argument("--output", default="scoreboard_detector.tflite")
    args = parser.parse_args()

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(args.output, "wb") as handle:
        handle.write(tflite_model)


if __name__ == "__main__":
    main()
