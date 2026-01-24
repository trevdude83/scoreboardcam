import argparse
import numpy as np
from PIL import Image
import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser(description="TFLite sanity check")
    parser.add_argument("--model", default="scoreboard_detector.tflite")
    parser.add_argument("--image", required=True)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = Image.open(args.image).convert("RGB").resize((args.img_size, args.img_size))
    x = np.array(image, dtype=np.float32)
    x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0][0]

    print(f"scoreboard probability: {output:.4f}")


if __name__ == "__main__":
    main()
