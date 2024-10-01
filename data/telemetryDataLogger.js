const axios = require("axios");
const createCsvWriter = require("csv-writer").createObjectCsvWriter;

// Set up CSV writer
const csvWriter = createCsvWriter({
  path: "careful-driving.csv",
  header: [
    { id: "time", title: "Time" },
    { id: "distance", title: "Distance" },
    { id: "speed", title: "speed" },
    { id: "acceleration_x", title: "AccelerationX" },
    { id: "acceleration_y", title: "AccelerationY" },
    { id: "acceleration_z", title: "AccelerationZ" },
    { id: "speed_limit", title: "SpeedLimit" },
    { id: "speed_difference", title: "Speed Difference" },
    { id: "speed_severity", title: "Speed Severity" },
  ],
});

// Custom time and distance tracking
let time = 0;
let lastUpdateTime = null;
let lastCustomTime = "";
let totalDistance = 0;
let lastSpeed = 0;

// Fetch data and write to CSV
const fetchData = async () => {
  try {
    const response = await axios.get(
      "http://localhost:25555/api/ets2/telemetry"
    );
    const data = response.data;

    // Check if the game is paused
    if (data.game.paused) {
      console.log("Game is paused. Paused collecting data");
      lastUpdateTime = null;
      return;
    }

    // Update custom time if the game is running
    const currentTime = Date.now();
    if (lastUpdateTime) {
      const deltaTime = currentTime - lastUpdateTime;

      time += deltaTime;

      const deltaTimeSeconds = deltaTime / 1000;

      const avgSpeed = (lastSpeed + data.truck.speed) / 2 / 3.6;

      const distanceTravelled = avgSpeed * deltaTimeSeconds;

      totalDistance += distanceTravelled;
    }

    lastUpdateTime = currentTime;

    if (time === lastCustomTime) {
      return;
    }

    lastCustomTime = time;

    let speed = data.truck.speed;
    if (speed > -0.5 && speed < 0.5) {
      speed = 0.0;
    }

    const speedLimit = data.navigation.speedLimit;
    const speedDifference = speed - speedLimit;
    const speedSeverity =
      speedLimit > 0 && speedDifference > 0
        ? Math.min(speedDifference / speedLimit, 1)
        : 0;

    lastSpeed = speed;

    const record = {
      time: (time / (1000 * 60)).toFixed(2),
      distance: (totalDistance / 1000).toFixed(8),
      speed: speed,
      acceleration_x: data.truck.acceleration.x,
      acceleration_y: data.truck.acceleration.y,
      acceleration_z: data.truck.acceleration.z,
      speed_limit: speedLimit,
      speed_difference: speedDifference.toFixed(2),
      speed_severity: speedSeverity.toFixed(2),
    };

    await csvWriter.writeRecords([record]);

    console.log("Data written to CSV...:", record);
  } catch (error) {
    console.error("Error fetching or writing data:", error);
  }
};

setInterval(fetchData, 1000);

console.log("Fetching data...");
