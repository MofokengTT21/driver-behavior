const { Tensor } = require("onnxruntime-node");
const axios = require("axios");
const createCsvWriter = require("csv-writer").createObjectCsvWriter;
const onnx = require("onnxruntime-node");
const express = require("express");
const path = require("path");
const fs = require("fs");
const cors = require("cors");

const app = express();
 
app.use(cors());
app.use(express.static(path.join(__dirname, "./ets2-telemetry-server/server/Html/skins/truck-monitor/")));

// Load scaler parameters for both models
const behaviorScalerParams = JSON.parse(
  fs.readFileSync(
    path.join(__dirname, "./ml-models/scaler/driver_behavior_scaler_params.json"),
    "utf8"
  )
);
const brakingScalerParams = JSON.parse(
  fs.readFileSync(
    path.join(__dirname, "./ml-models/scaler/braking_scaler_params.json"),
    "utf8"
  )
);
const behaviorMean = behaviorScalerParams.mean;
const behaviorVariance = behaviorScalerParams.var;
const brakingMean = brakingScalerParams.mean;
const brakingVariance = brakingScalerParams.var;

const csvWriter = createCsvWriter({
  path: "./predictions/latest-driver-behavior.csv",
  header: [
    { id: "time", title: "Time" },
    { id: "distance", title: "Distance" },
    { id: "speed", title: "Speed" },
    { id: "acceleration_x", title: "Acceleration X" },
    { id: "acceleration_y", title: "Acceleration Y" },
    { id: "acceleration_z", title: "Acceleration Z" },
    { id: "speed_limit", title: "Speed Limit" },
    { id: "speed_difference", title: "Speed Difference" },
    { id: "speed_severity", title: "Speed Severity" },
    { id: "behavior_prediction", title: "Behavior Prediction" },
    { id: "braking_prediction", title: "Braking Prediction" },
  ],
});

let modelSessionBehavior;
let modelSessionBraking;
let lastSpeed = 0;
let time = 0;
let totalDistance = 0;
let lastUpdateTime = null;
let lastCustomTime = 0;
let speedInRangeTime = 0;

const behaviorLabelMapping = {
  0: "aggressive",
  1: "rushed",
  2: "safe",
};

const brakingLabelMapping = {
  0: "harsh",
  1: "smooth",
};

// Load both ONNX models
Promise.all([
  onnx.InferenceSession.create(
    path.join(__dirname, "./ml-models/improved_lstm_model.onnx")
  ).then((session) => {
    modelSessionBehavior = session;
    console.log("Behavior ONNX model loaded successfully.");
  }),
  onnx.InferenceSession.create(
    path.join(__dirname, "./ml-models/improved_braking_lstm_model.onnx")
  ).then((session) => {
    modelSessionBraking = session;
    console.log("Braking ONNX model loaded successfully.");
  }),
]).catch((error) => {
  console.error("Error loading ONNX models:", error);
});

const bufferSize = 1;
const sequenceSize = 1;
let dataBuffer = [];

// Function to standardize data
function standardize(data, mean, variance) {
  return data.map((value, index) => {
    const scale = Math.sqrt(variance[index]);
    const shift = mean[index];

    if (scale === 0) return 0;
    if (isNaN(value) || isNaN(scale) || isNaN(shift)) {
      console.warn(`Warning: NaN detected in standardization.`);
      return 0;
    }
    return (value - shift) / scale;
  });
}

// Softmax function
function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((logit) => Math.exp(logit - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((exp) => exp / sumExps);
}

app.get("/sse", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  const fetchData = async () => {
    try {
      const response = await axios.get(
        "http://localhost:25555/api/ets2/telemetry"
      );
      const data = response.data;

      if (data.game.paused) {
        console.log("Game is paused. Paused data collection");
        return;
      }

      // Update time and distance
      const currentTime = Date.now();
      const deltaTime = lastUpdateTime ? (currentTime - lastUpdateTime) : 0;
      if (deltaTime > 0) {
        time += deltaTime;
        const avgSpeed = (lastSpeed + data.truck.speed) / 2 / 3.6;
        const distanceTravelled = avgSpeed * (deltaTime / 1000);
        totalDistance += distanceTravelled;
      }
      lastUpdateTime = currentTime;

      if (time === lastCustomTime) return;

      lastCustomTime = time;

      let speed = data.truck.speed;
      if (speed > -0.5 && speed < 0.5) {
        speedInRangeTime += deltaTime / 1000;
        speed = 0.0;
      } else {
        speedInRangeTime = 0; // Reset if speed is out of range
      }

      let record = {
        time: (time / (1000 * 60)).toFixed(2),
        distance: (totalDistance / 1000).toFixed(8),
        speed: speed,
        acceleration_x: data.truck.acceleration.x,
        acceleration_y: data.truck.acceleration.y,
        acceleration_z: data.truck.acceleration.z,
        speed_limit: data.navigation.speedLimit,
        speed_difference: 0,
        speed_severity: 0,
      };

      // Check if speed is low for 5 seconds
      if (speedInRangeTime >= 5) {
        record.prediction = "Not driving";
        dataBuffer = [];
        await csvWriter.writeRecords([record]);
        console.log("Data written to CSV without prediction due to low speed:", record);
        res.write(`data: ${JSON.stringify(record)}\n\n`);
        return;
      }

      // Calculate speed difference and severity
      const speedLimit = data.navigation.speedLimit;
      const speedDifference = speed - speedLimit;
      const speedSeverity =
        speedLimit > 0 && speedDifference > 0
          ? Math.min(speedDifference / speedLimit, 1)
          : 0;

      record.speed_difference = speedDifference.toFixed(2);
      record.speed_severity = speedSeverity.toFixed(2);

      dataBuffer.push([
        parseFloat(record.time),
        parseFloat(record.distance),
        parseFloat(record.speed),
        parseFloat(record.acceleration_x),
        parseFloat(record.acceleration_y),
        parseFloat(record.acceleration_z),
        parseFloat(record.speed_limit),
        parseFloat(record.speed_difference),
        parseFloat(record.speed_severity),
      ]);

      if (dataBuffer.length >= bufferSize) {
        // Predict driver behavior
        const standardizedBehaviorBuffer = dataBuffer.map((row) => standardize(row, behaviorMean, behaviorVariance));
        const behaviorInputArray = new Float32Array(standardizedBehaviorBuffer.flat());
        const behaviorInputTensor = new Tensor("float32", behaviorInputArray, [
          1,
          sequenceSize,
          9, // For 9 input features
        ]);

        // Predict braking
        const standardizedBrakingBuffer = dataBuffer.map((row) => standardize(row, brakingMean, brakingVariance));
        const brakingInputArray = new Float32Array(standardizedBrakingBuffer.flat());
        const brakingInputTensor = new Tensor("float32", brakingInputArray, [
          1,
          sequenceSize,
          9, // For 9 input features
        ]);

        const hiddenSize = 64;
        const h0 = new Tensor("float32", new Float32Array(2 * 1 * hiddenSize), [
          2,
          1,
          hiddenSize,
        ]);
        const c0 = new Tensor("float32", new Float32Array(2 * 1 * hiddenSize), [
          2,
          1,
          hiddenSize,
        ]);

        try {
          // Run behavior model
          const behaviorResults = await modelSessionBehavior.run({
            input: behaviorInputTensor,
            h0: h0,
            c0: c0,
          });
          const behaviorOutputArray = behaviorResults["output"].data;
          const behaviorProbabilities = softmax(behaviorOutputArray);
          const behaviorPredictionIndex = behaviorProbabilities.indexOf(
            Math.max(...behaviorProbabilities)
          );
          record.behavior_prediction = behaviorLabelMapping[behaviorPredictionIndex] || "unknown";

          // Run braking model
          const brakingResults = await modelSessionBraking.run({
            input: brakingInputTensor,
            h0: h0,
            c0: c0,
          });
          const brakingOutputArray = brakingResults["output"].data;
          const brakingProbabilities = softmax(brakingOutputArray);
          const brakingPredictionIndex = brakingProbabilities.indexOf(
            Math.max(...brakingProbabilities)
          );
          record.braking_prediction = brakingLabelMapping[brakingPredictionIndex] || "unknown";
        } catch (error) {
          console.error("Error in ONNX inference:", error);
        }

        // Clear buffer after processing
        dataBuffer = [];
      }

      await csvWriter.writeRecords([record]);
      console.log("Data written to CSV:", record);
      res.write(`data: ${JSON.stringify(record)}\n\n`);
    } catch (error) {
      console.error("Error fetching telemetry data:", error);
    }
  };

  setInterval(fetchData, 1000);
});

const PORT = 8080;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
