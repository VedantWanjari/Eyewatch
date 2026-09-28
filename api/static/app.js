const startCameraButton = document.getElementById("start-camera");
const captureButton = document.getElementById("capture-detect");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

let cameraStream = null;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#b91c1c" : "#1f2937";
}

function renderDetections(detections) {
  resultsEl.innerHTML = "";
  detections.forEach((detection) => {
    const li = document.createElement("li");
    li.textContent = `${detection.class_name} (id=${detection.class_id}) confidence=${detection.confidence.toFixed(3)} bbox=[${detection.bbox.map((v) => v.toFixed(1)).join(", ")}]`;
    resultsEl.appendChild(li);
  });
}

function drawBoxes(detections) {
  const ctx = canvas.getContext("2d");
  ctx.lineWidth = 2;
  ctx.font = "14px Arial";
  detections.forEach((detection) => {
    const [x1, y1, x2, y2] = detection.bbox;
    const width = x2 - x1;
    const height = y2 - y1;
    ctx.strokeStyle = "#22c55e";
    ctx.fillStyle = "#22c55e";
    ctx.strokeRect(x1, y1, width, height);
    ctx.fillText(
      `${detection.class_name} ${(detection.confidence * 100).toFixed(1)}%`,
      x1,
      Math.max(14, y1 - 6),
    );
  });
}

startCameraButton.addEventListener("click", async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    video.srcObject = cameraStream;
    captureButton.disabled = false;
    setStatus("Camera ready.");
  } catch (error) {
    setStatus(`Unable to access camera: ${error.message}`, true);
  }
});

captureButton.addEventListener("click", async () => {
  if (!cameraStream) {
    setStatus("Enable camera first.", true);
    return;
  }

  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) {
    setStatus("Video feed is not ready yet.", true);
    return;
  }

  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, width, height);

  setStatus("Running detection...");
  captureButton.disabled = true;
  resultsEl.innerHTML = "";

  try {
    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.92),
    );
    if (!blob) {
      throw new Error("Failed to capture image from camera frame");
    }
    const formData = new FormData();
    formData.append("image", blob, "capture.jpg");

    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Inference request failed");
    }

    drawBoxes(data.detections || []);
    if (data.count === 0) {
      setStatus("No detections.");
    } else {
      setStatus(`Detected ${data.count} object(s).`);
    }
    renderDetections(data.detections || []);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    captureButton.disabled = false;
  }
});
