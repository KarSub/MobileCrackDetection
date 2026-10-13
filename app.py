
import os
import io
import time
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

def classify_crack(mask):
    binary_mask = np.asarray(mask) > 0

    if not np.any(binary_mask):
        return 0.0, "No crack detected", "No action required"

    skeleton = skeletonize(binary_mask)
    distances = distance_transform_edt(binary_mask)

    if not np.any(skeleton):
        return 0.0, "No crack detected", "Check the predicted mask"

    avg_width = float(np.mean(distances[skeleton]) * 2)

    if avg_width < 8.45:
        severity = "Low"
        action = "Sealing"
    elif avg_width <= 18.84:
        severity = "Medium"
        action = "Patching"
    else:
        severity = "High"
        action = "Reconstruction"

    return avg_width, severity, action


st.set_page_config(
    page_title="CrackVision | BiSeNetV2",
    page_icon="🔍",
    layout="wide"
)
MODEL_PATH = "bisenetv2.keras"
INPUT_SIZE = (512, 512)
THRESHOLD = 0.77397


@st.cache_resource
def load_crack_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


def get_prediction(model, image):
    original = image.convert("RGB")
    original_array = np.array(original)

    resized = original.resize(
        (INPUT_SIZE[1], INPUT_SIZE[0]),
        Image.Resampling.BILINEAR
    )

    input_array = np.asarray(resized, dtype=np.float32)

    mean = np.mean(input_array)
    std = np.std(input_array)
    input_array = (input_array - mean) / (std + 1e-7)
    input_array = np.expand_dims(input_array, axis=0)

    start = time.perf_counter()
    output = model(input_array, training=False)
    elapsed = (time.perf_counter() - start) * 1000

    if isinstance(output, dict):
        output = next(iter(output.values()))
    elif isinstance(output, (list, tuple)):
        output = output[0]

    output = np.asarray(output)

    if output.ndim == 4:
        output = output[0]

    if output.ndim == 2:
        probability = output
    elif output.ndim == 3 and output.shape[-1] == 1:
        probability = output[..., 0]
    elif output.ndim == 3 and output.shape[-1] >= 2:
        probability = output[..., 1]
    else:
        raise ValueError(
            f"Unexpected model output shape: {output.shape}"
        )

    st.write("Prediction output shape:", output.shape)
    st.write("Minimum probability:", float(probability.min()))
    st.write("Maximum probability:", float(probability.max()))
    st.write("Mean probability:", float(probability.mean()))

    mask = probability >= THRESHOLD

    mask_image = Image.fromarray(
        (mask.astype(np.uint8) * 255)
    )
    mask_image = mask_image.resize(
        original.size,
        Image.Resampling.NEAREST
    )

    mask_array = np.asarray(mask_image) > 0

    overlay = original_array.copy()
    overlay[mask_array] = (
        0.65 * overlay[mask_array] +
        0.35 * np.array([255, 0, 0])
    ).astype(np.uint8)

    overlay_image = Image.fromarray(overlay)
    mask_display = Image.fromarray(
        mask_array.astype(np.uint8) * 255
    )

    return mask_display, overlay_image, elapsed, mask_array

st.title("🔍 CrackVision")
st.subheader("Concrete Crack Segmentation using BiSeNetV2")
st.write(
    "Capture or upload a concrete surface image to "
    "visualize the cracks detected by the trained model."
)

st.info("Inference runs on the connected computer, not on the phone.")

try:
    model = load_crack_model()
except Exception as error:
    st.error("The model could not be loaded.")
    st.code(str(error))
    st.warning(
        "Check the model path, whether the file contains the "
        "complete model, and whether custom layers are required."
    )
    st.stop()

st.success("BiSeNetV2 model loaded successfully.")

camera_image = st.camera_input("Take a concrete surface photo")
uploaded_image = st.file_uploader(
    "Or choose an image from your gallery",
    type=["jpg", "jpeg", "png"]
)

source = camera_image if camera_image is not None else uploaded_image

if source is not None:
    image = Image.open(source).convert("RGB")

    st.subheader("Input image")
    st.image(image, use_container_width=True)

    
if st.button("Detect cracks", type="primary", use_container_width=True):
    try:
        with st.spinner("Running crack segmentation..."):
            mask, overlay, elapsed, mask_array = get_prediction(
                model, image
            )

        avg_width, severity, action = classify_crack(mask_array)

        st.subheader("Crack Severity Classification")
        st.metric("Average Crack Width", f"{avg_width:.2f} px")
        st.metric("Severity Level", severity)
        st.write(f"**Recommended Maintenance:** {action}")

        st.subheader("Segmentation results")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                mask,
                caption="Predicted crack mask",
                use_container_width=True
            )

        with col2:
            st.image(
                overlay,
                caption="Cracks highlighted in red",
                use_container_width=True
            )

        crack_percentage = (
            np.count_nonzero(mask_array) /
            mask_array.size
        ) * 100

        st.metric("Inference time", f"{elapsed:.1f} ms")
        st.metric(
            "Image pixels classified as crack",
            f"{crack_percentage:.2f}%"
        )

        st.caption(
            "The crack percentage is the fraction of pixels "
            "classified as crack, not a measurement of physical "
            "crack width or structural safety."
        )

        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")

        st.download_button(
            "Download overlay",
            data=buffer.getvalue(),
            file_name="crack_overlay.png",
            mime="image/png",
            use_container_width=True
        )

    except Exception as error:
        st.error(
            "Prediction failed. Check your model output and preprocessing."
        )
        st.code(str(error))
